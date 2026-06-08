module KernelInterpolationCellAverageExt

using LinearAlgebra: Symmetric, norm
using Meshes: Meshes, Box, Point, measure, to, ustrip, integral, centroid, boundingbox
using RecipesBase: @recipe, @series

if pkgversion(Meshes) < v"0.57"
    error("""
    KernelInterpolationCellAverageExt requires Meshes ≥ 0.57.
    Please upgrade: ] update Meshes
    """)
end

using KernelInterpolation: KernelInterpolation

# Meshes stores coordinates as Unitful quantities; strip units to get plain Float64 SVectors.
function _to_coords(p::Point)
    return ustrip.(to(p))
end

# ── CellAverageFunctional constructor ─────────────────────────────────────────

# Works for any parametrized Meshes.jl geometry; measure() gives the exact volume.
function KernelInterpolation.CellAverageFunctional(volume::Meshes.Geometry)
    vol_measure = ustrip(measure(volume))
    return KernelInterpolation.CellAverageFunctional{Meshes.embeddim(volume)}(
        volume, vol_measure)
end

# ── Matrix assembly ───────────────────────────────────────────────────────────

# Approximates A[i,j] = (1/|Ωᵢ||Ωⱼ|) ∫_{Ωᵢ} ∫_{Ωⱼ} K(x,y) dy dx via nested
# h-adaptive integration. The inner integral is recomputed at every outer quadrature
# point; this is correct for any geometry but slow for large N.
# ustrip is required because Meshes.integral returns a Unitful quantity (m^d × f-units);
# volume_measure is already stripped, so we must strip the integrals to get plain Float64.
function _entry(func_i, func_j, kernel)
    inner(p) = ustrip(integral(q -> kernel(_to_coords(p), _to_coords(q)), func_j.volume))
    return ustrip(integral(p -> inner(p), func_i.volume)) /
           (func_i.volume_measure * func_j.volume_measure)
end

function KernelInterpolation.assemble_cell_average_matrix(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        kernel::KernelInterpolation.AbstractKernel)
    n = length(functionals)
    A = Matrix{Float64}(undef, n, n)
    n_entries = n * (n + 1) ÷ 2
    @info "Assembling $(n)×$(n) cell-average matrix ($n_entries entries, upper triangle)"
    t_start = time()
    entries_done = 0
    # Exploit kernel symmetry K(x,y) = K(y,x): compute upper triangle only.
    for i in 1:n
        t_row = time()
        for j in i:n
            A[i, j] = _entry(functionals[i], functionals[j], kernel)
            A[j, i] = A[i, j]
        end
        entries_done += n - i + 1
        pct = round(Int, 100 * entries_done / n_entries)
        @info "  row $i/$n  ($pct%,  $(round(time() - t_row; digits = 1))s/row,  $(round(time() - t_start; digits = 1))s total)"
    end
    @info "Matrix assembly complete  ($(round(time() - t_start; digits = 1))s)"
    return A
end

function KernelInterpolation.cell_average_interpolate(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        values::AbstractVector{<:Real},
        kernel::KernelInterpolation.AbstractKernel;
        linsolve = nothing)
    n = length(functionals)
    @assert length(values) == n "number of values must match number of functionals"
    A     = KernelInterpolation.assemble_cell_average_matrix(functionals, kernel)
    # Wrap as Symmetric so the solver can use Cholesky; kernel Gram matrices are SPD.
    # Independent adaptive evaluation of A[i,j] and A[j,i] may differ by ~quadrature
    # tolerance — Symmetric picks the upper triangle as the authoritative value.
    A_sym = Symmetric(A)
    c     = KernelInterpolation.solve_linear_system(A_sym, Float64.(values), linsolve)
    return KernelInterpolation.CellAverageInterpolation(kernel, functionals, c, A_sym)
end

# ── Interpolant evaluation ────────────────────────────────────────────────────

# s(x) = Σⱼ cⱼ ψⱼ(x)  where  ψⱼ(x) = (1/|Ωⱼ|) ∫_{Ωⱼ} K(x,y) dy.
function (itp::KernelInterpolation.CellAverageInterpolation)(x::AbstractVector)
    s = zero(Float64)
    for (j, func) in enumerate(itp.functionals)
        s += itp.c[j] *
             ustrip(integral(q -> itp.kernel(x, _to_coords(q)), func.volume)) /
             func.volume_measure
    end
    return s
end

function (itp::KernelInterpolation.CellAverageInterpolation{1})(x::Real)
    return itp([x])
end

# ── Cell-average recovery ─────────────────────────────────────────────────────

# A*c recovers the cell averages of s exactly (λᵢ(s) = (Ac)ᵢ by construction),
# so no integration is needed here.
_algebraic_avg(itp) = KernelInterpolation.system_matrix(itp) *
                      KernelInterpolation.coefficients(itp)

function KernelInterpolation.cell_averages(itp::KernelInterpolation.CellAverageInterpolation)
    return _algebraic_avg(itp)
end

# ── Geometry utilities ────────────────────────────────────────────────────────

# Use the bounding box so _bounds works for any geometry, not just Box.
function _bounds(func::KernelInterpolation.CellAverageFunctional)
    bb = boundingbox(func.volume)
    return _to_coords(bb.min), _to_coords(bb.max)
end

function KernelInterpolation.mesh_diameter(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional})
    return maximum(functionals) do func
        lo, hi = _bounds(func)
        norm(hi .- lo)
    end
end

function KernelInterpolation.centroid_nodeset(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional})
    coords = [_to_coords(centroid(func.volume)) for func in functionals]
    return KernelInterpolation.NodeSet(coords)
end

# ── Visualization helpers ─────────────────────────────────────────────────────

function _domain_1d(funcs)
    lo = minimum(_bounds(func)[1][1] for func in funcs)
    hi = maximum(_bounds(func)[2][1] for func in funcs)
    return lo, hi
end

function _domain_2d(funcs)
    lo_x = minimum(_bounds(func)[1][1] for func in funcs)
    hi_x = maximum(_bounds(func)[2][1] for func in funcs)
    lo_y = minimum(_bounds(func)[1][2] for func in funcs)
    hi_y = maximum(_bounds(func)[2][2] for func in funcs)
    return lo_x, hi_x, lo_y, hi_y
end

function _step_xy_1d(funcs, vals)
    xs = Float64[]
    ys = Float64[]
    for (func, v) in zip(funcs, vals)
        lo, hi = _bounds(func)
        append!(xs, (lo[1], hi[1], NaN))
        append!(ys, (v,     v,     NaN))
    end
    return xs, ys
end

function _rasterize_2d(funcs, vals, x, y)
    z = fill(NaN, length(y), length(x))
    # O(pixels × cells) point-in-geometry scan; acceptable for visualization grids (≤200²).
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

# ── Plot recipes ──────────────────────────────────────────────────────────────

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

@recipe function f(itp::KernelInterpolation.CellAverageInterpolation{2};
                   x_min = nothing, x_max = nothing,
                   y_min = nothing, y_max = nothing, N = 50)
    funcs                   = KernelInterpolation.functionals(itp)
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

@recipe function f(itp::KernelInterpolation.CellAverageInterpolation{2},
                   target::Function;
                   x_min = nothing, x_max = nothing,
                   y_min = nothing, y_max = nothing, N = 50)
    funcs                   = KernelInterpolation.functionals(itp)
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
