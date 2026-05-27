module KernelInterpolationCellAverageExt

using LinearAlgebra: Symmetric, norm
using Meshes: Meshes, Point, measure, integral, to, ustrip, centroid
using RecipesBase: @recipe, @series

if pkgversion(Meshes) < v"0.57"
    error("""
    KernelInterpolationCellAverageExt requires Meshes ≥ 0.57.
    Please upgrade: ] update Meshes
    """)
end

using KernelInterpolation: KernelInterpolation

# Helper function to strip units from Meshes.jl Points
function _to_coords(p::Point)
    return ustrip.(to(p))
end

# Override the constructor to accept any Meshes geometry
# Using the abstract Geometry type to capture all geometries
function KernelInterpolation.CellAverageFunctional(volume)
    vol_measure = ustrip(measure(volume))
    return KernelInterpolation.CellAverageFunctional{Meshes.embeddim(volume)}(volume, vol_measure)
end

"""
    assemble_cell_average_matrix(functionals::Vector{CellAverageFunctional}, 
                                 kernel::AbstractKernel;
                                 ibackend=nothing, 
                                 dbackend=nothing)

Assemble the kernel matrix for cell-average functionals.

For cell-average functionals λᵢ and λⱼ, the matrix entry is:
    A_ij = (1/|Vᵢ||Vⱼ|) ∫_Vᵢ ∫_Vⱼ K(x, y) dy dx

The double integral is computed using Meshes.jl's integral function via nested quadrature.
"""
function KernelInterpolation.assemble_cell_average_matrix(
    functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
    kernel::KernelInterpolation.AbstractKernel;
    ibackend=nothing,
    dbackend=nothing)
    
    n = length(functionals)
    A = Matrix{Float64}(undef, n, n)
    
    for i in 1:n
        for j in 1:n
            A[i, j] = _compute_cell_average_kernel_entry(
                functionals[i], functionals[j], kernel, ibackend, dbackend
            )
        end
    end
    
    return A
end

function _compute_cell_average_kernel_entry(
    func_i::KernelInterpolation.CellAverageFunctional,
    func_j::KernelInterpolation.CellAverageFunctional,
    kernel::KernelInterpolation.AbstractKernel,
    ibackend, dbackend)
    
    V_i = func_i.volume
    V_j = func_j.volume
    measure_i = func_i.volume_measure
    measure_j = func_j.volume_measure
    
    # Inner integral: ∫_Vⱼ K(x, y) dy for a fixed x
    function inner_integral_wrapper(x)
        x_vec = _to_coords(x)

        function integrand_inner(y)
            y_vec = _to_coords(y)
            return kernel(x_vec, y_vec)
        end
        
        if ibackend !== nothing && dbackend !== nothing
            return integral(integrand_inner, V_j; ibackend=ibackend, dbackend=dbackend)
        else
            return integral(integrand_inner, V_j)
        end
    end
    
    # Outer integral: ∫_Vᵢ (∫_Vⱼ K(x, y) dy) dx
    if ibackend !== nothing && dbackend !== nothing
        result = integral(inner_integral_wrapper, V_i; ibackend=ibackend, dbackend=dbackend)
    else
        result = integral(inner_integral_wrapper, V_i)
    end

    # Normalize by both volume measures (ustrip removes Unitful units from the integral result)
    return ustrip(result) / (measure_i * measure_j)
end

function KernelInterpolation.cell_average_interpolate(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        values::AbstractVector{<:Real},
        kernel::KernelInterpolation.AbstractKernel;
        ibackend = nothing,
        dbackend = nothing,
        linsolve = nothing)
    n = length(functionals)
    @assert length(values) == n "number of values must match number of functionals"

    A = KernelInterpolation.assemble_cell_average_matrix(functionals, kernel;
                                                          ibackend = ibackend,
                                                          dbackend = dbackend)
    A_sym = Symmetric(A)
    c = KernelInterpolation.solve_linear_system(A_sym, Float64.(values), linsolve)

    return KernelInterpolation.CellAverageInterpolation(kernel, functionals, c, A_sym)
end

function (itp::KernelInterpolation.CellAverageInterpolation)(x::AbstractVector)
    s = zero(eltype(itp.c))
    for (j, func) in enumerate(itp.functionals)
        # λⱼ(K(x, ·)) = (1/|Vⱼ|) ∫_Vⱼ K(x, y) dy
        integrand = y -> itp.kernel(x, _to_coords(y))
        s += itp.c[j] * ustrip(integral(integrand, func.volume)) / func.volume_measure
    end
    return s
end

function (itp::KernelInterpolation.CellAverageInterpolation{1})(x::Real)
    return itp([x])
end

function KernelInterpolation.cell_averages(itp::KernelInterpolation.CellAverageInterpolation)
    result = Vector{Float64}(undef, length(itp.functionals))
    for (i, func) in enumerate(itp.functionals)
        integrand = p -> itp(_to_coords(p))
        result[i] = ustrip(integral(integrand, func.volume)) / func.volume_measure
    end
    return result
end

function KernelInterpolation.mesh_diameter(functionals::Vector{<:KernelInterpolation.CellAverageFunctional})
    return maximum(functionals) do func
        lo, hi = _to_coords(func.volume.min), _to_coords(func.volume.max)
        norm(hi .- lo)
    end
end

function KernelInterpolation.centroid_nodeset(functionals::Vector{<:KernelInterpolation.CellAverageFunctional})
    coords = [_to_coords(centroid(func.volume)) for func in functionals]
    return KernelInterpolation.NodeSet(coords)
end

# ── visualization helpers ────────────────────────────────────────────────────

# Min/max corner coordinates of a box-shaped functional
function _bounds(func::KernelInterpolation.CellAverageFunctional)
    return _to_coords(func.volume.min), _to_coords(func.volume.max)
end

# Infer 1D domain from a vector of functionals
function _domain_1d(funcs)
    lo = minimum(_bounds(func)[1][1] for func in funcs)
    hi = maximum(_bounds(func)[2][1] for func in funcs)
    return lo, hi
end

# Infer 2D domain from a vector of functionals
function _domain_2d(funcs)
    lo_x = minimum(_bounds(func)[1][1] for func in funcs)
    hi_x = maximum(_bounds(func)[2][1] for func in funcs)
    lo_y = minimum(_bounds(func)[1][2] for func in funcs)
    hi_y = maximum(_bounds(func)[2][2] for func in funcs)
    return lo_x, hi_x, lo_y, hi_y
end

# NaN-separated (x, y) coordinates for a 1D step-function plot
function _step_xy_1d(funcs, vals)
    xs = Float64[]
    ys = Float64[]
    for (func, v) in zip(funcs, vals)
        lo, hi = _bounds(func)
        append!(xs, (lo[1], hi[1], NaN))
        append!(ys, (v,    v,    NaN))
    end
    return xs, ys
end

# Rasterize 2D cell averages onto a fine (x, y) grid via point-in-cell test
function _rasterize_2d(funcs, vals, x, y)
    z = fill(NaN, length(y), length(x))
    for (ix, xv) in enumerate(x), (iy, yv) in enumerate(y)
        p = Point(xv, yv)
        for (func, v) in zip(funcs, vals)
            if p ∈ func.volume
                z[iy, ix] = v
                break
            end
        end
    end
    return z
end

# Cell averages recovered algebraically from A * c (exact, no quadrature)
_algebraic_avg(itp) = KernelInterpolation.system_matrix(itp) *
                      KernelInterpolation.coefficients(itp)

# ── recipes ─────────────────────────────────────────────────────────────────

# 1D: step function of cell averages + smooth interpolant
@recipe function f(itp::KernelInterpolation.CellAverageInterpolation{1};
                   x_min = nothing, x_max = nothing, N = 200)
    funcs  = KernelInterpolation.functionals(itp)
    lo, hi = _domain_1d(funcs)
    lo     = @something(x_min, lo)
    hi     = @something(x_max, hi)
    x      = collect(LinRange(lo, hi, N))

    @series begin
        xs, ys    = _step_xy_1d(funcs, _algebraic_avg(itp))
        label     --> "cell averages"
        linestyle --> :dash
        linewidth --> 2
        xs, ys
    end
    @series begin
        label  --> "interpolant s(x)"
        xguide --> "x"
        yguide --> "f"
        x, itp.(x)
    end
end

# 1D: target function + step function + smooth interpolant
@recipe function f(itp::KernelInterpolation.CellAverageInterpolation{1},
                   target::Function;
                   x_min = nothing, x_max = nothing, N = 200)
    funcs  = KernelInterpolation.functionals(itp)
    lo, hi = _domain_1d(funcs)
    lo     = @something(x_min, lo)
    hi     = @something(x_max, hi)
    x      = collect(LinRange(lo, hi, N))

    @series begin
        label --> "target f(x)"
        x, target.(x)
    end
    @series begin
        xs, ys    = _step_xy_1d(funcs, _algebraic_avg(itp))
        label     --> "cell averages λᵢ(f)"
        linestyle --> :dash
        linewidth --> 2
        xs, ys
    end
    @series begin
        label  --> "interpolant s(x)"
        xguide --> "x"
        yguide --> "f"
        x, itp.(x)
    end
end

# 2D: smooth interpolant as heatmap
@recipe function f(itp::KernelInterpolation.CellAverageInterpolation{2};
                   x_min = nothing, x_max = nothing,
                   y_min = nothing, y_max = nothing, N = 50)
    funcs              = KernelInterpolation.functionals(itp)
    lo_x, hi_x, lo_y, hi_y = _domain_2d(funcs)
    lo_x = @something(x_min, lo_x); hi_x = @something(x_max, hi_x)
    lo_y = @something(y_min, lo_y); hi_y = @something(y_max, hi_y)

    x = collect(LinRange(lo_x, hi_x, N))
    y = collect(LinRange(lo_y, hi_y, N))

    seriestype --> :heatmap
    xguide     --> "x"
    yguide     --> "y"
    x, y, [itp([xv, yv]) for yv in y, xv in x]
end

# 2D: piecewise-constant heatmap of cell averages + contour lines of interpolant
@recipe function f(itp::KernelInterpolation.CellAverageInterpolation{2},
                   target::Function;
                   x_min = nothing, x_max = nothing,
                   y_min = nothing, y_max = nothing, N = 50)
    funcs              = KernelInterpolation.functionals(itp)
    lo_x, hi_x, lo_y, hi_y = _domain_2d(funcs)
    lo_x = @something(x_min, lo_x); hi_x = @something(x_max, hi_x)
    lo_y = @something(y_min, lo_y); hi_y = @something(y_max, hi_y)

    x = collect(LinRange(lo_x, hi_x, N))
    y = collect(LinRange(lo_y, hi_y, N))

    @series begin
        z         = _rasterize_2d(funcs, _algebraic_avg(itp), x, y)
        seriestype := :heatmap
        xguide    --> "x"
        yguide    --> "y"
        label     --> "cell averages λᵢ(f)"
        x, y, z
    end
    @series begin
        z          = [itp([xv, yv]) for yv in y, xv in x]
        seriestype := :contour
        label      --> "interpolant s(x,y)"
        colorbar   --> false
        linewidth  --> 2
        x, y, z
    end
end

end
