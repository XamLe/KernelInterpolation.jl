module KernelInterpolationCellAverageExt

using LinearAlgebra: Symmetric, norm
using StaticArrays: SVector
using Meshes: Meshes, Box, Point, measure, to, ustrip
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

# ── Quadrature: uniform midpoint rule on a Box ────────────────────────────────
#
# Covers Box with n^dim midpoints on a uniform grid.
# Node at multi-index (i₁,…,i_d) (0-based): lo_d + (i_d + 0.5) * (hi_d - lo_d) / n.
# All weights equal: vol / n^dim, so sum(weights) = vol.

function _box_quadrature(volume::Box, n::Int)
    dim     = Meshes.embeddim(volume)
    lo      = _to_coords(volume.min)
    hi      = _to_coords(volume.max)
    vol     = ustrip(measure(volume))
    n_pts   = n^dim
    nodes   = Vector{SVector{dim, Float64}}(undef, n_pts)
    # Equal weights: each node represents a sub-cell of volume vol/n_pts, so
    # sum(weights) = vol and the weighted sum approximates the integral over the box.
    weight  = vol / n_pts
    k = 0
    # Iterate over all d-dimensional multi-indices via a Cartesian product of 0:n-1
    # ranges, producing the n^dim midpoints of the uniform sub-cell grid.
    for idx in Iterators.product(ntuple(_ -> 0:n-1, dim)...)
        k += 1
        coords = ntuple(d -> lo[d] + (idx[d] + 0.5) * (hi[d] - lo[d]) / n, dim)
        nodes[k] = SVector{dim, Float64}(coords...)
    end
    return nodes, fill(weight, n_pts)
end

# ── CellAverageFunctional constructor ─────────────────────────────────────────

function KernelInterpolation.CellAverageFunctional(volume::Box; n_quadrature::Int = 10)
    vol_measure = ustrip(measure(volume))
    quad        = _box_quadrature(volume, n_quadrature)
    return KernelInterpolation.CellAverageFunctional{Meshes.embeddim(volume)}(
        volume, vol_measure, quad)
end

# ── Matrix assembly ───────────────────────────────────────────────────────────

# Function barrier so Julia specialises on the concrete types of quad_i / quad_j,
# eliminating the type instability that arises from quadrature::Any.
function _quad_double_sum(quad_i, quad_j, mi, mj, kernel)
    nodes_i, w_i = quad_i
    nodes_j, w_j = quad_j
    s = 0.0
    @inbounds for k in eachindex(w_i)
        xi = nodes_i[k]
        wi = w_i[k]
        for l in eachindex(w_j)
            s += wi * w_j[l] * kernel(xi, nodes_j[l])
        end
    end
    # Dividing by both measures converts the double integral ∬K dx dy into
    # the cell-average inner product: λᵢ(λⱼ(K)) = (1/|Ωᵢ||Ωⱼ|) ∬ K(x,y) dx dy.
    return s / (mi * mj)
end

# Approximates the (i,j) entry of the Gram matrix A_{ij} = λᵢ(ψⱼ),
# where ψⱼ(x) = (1/|Ωⱼ|) ∫_{Ωⱼ} K(x,y) dy is the j-th basis functional applied to the kernel.
function _entry_quad(func_i, func_j, kernel)
    _quad_double_sum(func_i.quadrature, func_j.quadrature,
                     func_i.volume_measure, func_j.volume_measure, kernel)
end

function KernelInterpolation.assemble_cell_average_matrix(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        kernel::KernelInterpolation.AbstractKernel)
    n = length(functionals)
    A = Matrix{Float64}(undef, n, n)
    # Full n×n loop rather than exploiting kernel symmetry to keep the code simple;
    # symmetrisation is deferred to cell_average_interpolate via Symmetric(A).
    for i in 1:n
        for j in 1:n
            A[i, j] = _entry_quad(functionals[i], functionals[j], kernel)
        end
    end
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
    # Wrap as Symmetric so that the linear solver can use a Cholesky factorisation
    # rather than LU; the underlying kernel Gram matrix is symmetric positive definite
    # by the reproducing kernel property.
    A_sym = Symmetric(A)
    c     = KernelInterpolation.solve_linear_system(A_sym, Float64.(values), linsolve)
    return KernelInterpolation.CellAverageInterpolation(kernel, functionals, c, A_sym)
end

# ── Interpolant evaluation ────────────────────────────────────────────────────

# Approximates ∫_{Ωⱼ} K(x, y) dy for a fixed evaluation point x.
function _quad_single_sum(quad, x, kernel)
    nodes, w = quad
    s = 0.0
    @inbounds for k in eachindex(w)
        s += w[k] * kernel(x, nodes[k])
    end
    return s
end

# Evaluates s(x) = Σⱼ cⱼ ψⱼ(x)  where  ψⱼ(x) = (1/|Ωⱼ|) ∫_{Ωⱼ} K(x,y) dy
# is the j-th basis function (the cell-average functional applied to the kernel row).
function (itp::KernelInterpolation.CellAverageInterpolation)(x::AbstractVector)
    s = zero(eltype(itp.c))
    for (j, func) in enumerate(itp.functionals)
        s += itp.c[j] * _quad_single_sum(func.quadrature, x, itp.kernel) /
             func.volume_measure
    end
    return s
end

function (itp::KernelInterpolation.CellAverageInterpolation{1})(x::Real)
    return itp([x])
end

# ── Cell-average recovery ─────────────────────────────────────────────────────

function _quad_avg(quad, itp, volume_measure)
    nodes, w = quad
    s = 0.0
    @inbounds for k in eachindex(w)
        s += w[k] * itp(nodes[k])
    end
    return s / volume_measure
end

function KernelInterpolation.cell_averages(itp::KernelInterpolation.CellAverageInterpolation)
    return [_quad_avg(func.quadrature, itp, func.volume_measure)
            for func in itp.functionals]
end

# ── Geometry utilities ────────────────────────────────────────────────────────

function _bounds(func::KernelInterpolation.CellAverageFunctional)
    return _to_coords(func.volume.min), _to_coords(func.volume.max)
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
    coords = [(_to_coords(func.volume.min) .+ _to_coords(func.volume.max)) ./ 2
              for func in functionals]
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
    # O(pixels × cells) point-in-box scan; acceptable for visualization grids (≤200²).
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

# A*c recovers the cell averages of s: λᵢ(s) = Σⱼ cⱼ A_{ij} = (Ac)ᵢ.
# By construction this equals the interpolation data, but recomputing via A*c avoids
# a second quadrature pass and is used as the ground-truth in plot recipes.
_algebraic_avg(itp) = KernelInterpolation.system_matrix(itp) *
                      KernelInterpolation.coefficients(itp)

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
