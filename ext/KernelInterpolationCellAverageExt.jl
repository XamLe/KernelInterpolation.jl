module KernelInterpolationCellAverageExt

using LinearAlgebra: Symmetric, norm, cond
using Meshes: Meshes, Box, Point, Triangle, Polytope, measure, to, ustrip, integral,
              centroid, vertices, boundingbox, RegularGrid, elements, nelements, element,
              tesselate, DelaunayTesselation, VoronoiTesselation, PointSet
using RecipesBase: @recipe, @series
using FastGaussQuadrature
using IntegrationInterface

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
    inner(p) = ustrip(integral(q -> kernel(_to_coords(p), _to_coords(q)), func_j.volume;
                               # ibackend = Backend.Quadrature(gausslegendre(10))
                               ))
    return ustrip(integral(p -> inner(p), func_i.volume;
                           # ibackend = Backend.Quadrature(gausslegendre(10))
                                                         )) /
           (func_i.volume_measure * func_j.volume_measure)
end

function KernelInterpolation.assemble_cell_average_matrix(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        kernel::KernelInterpolation.AbstractKernel,
        RealT::Type{<:Real} = Float64)
    n         = length(functionals)
    A         = Matrix{RealT}(undef, n, n)
    n_entries = n * (n + 1) ÷ 2
    # BigFloat uses MPFR task-local state; sequential loop avoids pool contention.
    use_threads = !(RealT <: BigFloat)
    @info "Assembling $(n)×$(n) cell-average matrix ($n_entries entries, upper triangle, $(use_threads ? Threads.nthreads() : 1) thread(s), RealT=$RealT)"
    # Exploit kernel symmetry K(x,y) = K(y,x): compute upper triangle only.
    t = @elapsed if use_threads
        Threads.@threads for i in 1:n
            for j in i:n
                A[i, j] = RealT(_entry(functionals[i], functionals[j], kernel))
                A[j, i] = A[i, j]
            end
        end
    else
        for i in 1:n
            for j in i:n
                A[i, j] = RealT(_entry(functionals[i], functionals[j], kernel))
                A[j, i] = A[i, j]
            end
        end
    end
    # For BigFloat, computing cond via SVD in BigFloat is O(n³) at high precision;
    # a Float64 estimate is fast and sufficient for logging.
    κ = cond(RealT <: BigFloat ? Float64.(A) : A)
    @info "Matrix assembly complete  ($(round(t; digits = 1))s)  cond(A) ≈ $(round(κ; sigdigits = 4))"
    return A
end

function KernelInterpolation.cell_average_interpolate(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional},
        values::AbstractVector{RealT},
        kernel::KernelInterpolation.AbstractKernel;
        linsolve = nothing) where {RealT <: Real}
    n = length(functionals)
    @assert length(values) == n "number of values must match number of functionals"
    A     = KernelInterpolation.assemble_cell_average_matrix(functionals, kernel, RealT)
    # Wrap as Symmetric so the solver can use Cholesky; kernel Gram matrices are SPD.
    # Independent adaptive evaluation of A[i,j] and A[j,i] may differ by ~quadrature
    # tolerance — Symmetric picks the upper triangle as the authoritative value.
    A_sym = Symmetric(A)
    c     = KernelInterpolation.solve_linear_system(A_sym, RealT.(values), linsolve)
    return KernelInterpolation.CellAverageInterpolation(kernel, functionals, c, A_sym)
end

# ── Interpolant evaluation ────────────────────────────────────────────────────

# s(x) = Σⱼ cⱼ ψⱼ(x)  where  ψⱼ(x) = (1/|Ωⱼ|) ∫_{Ωⱼ} K(x,y) dy.
function (itp::KernelInterpolation.CellAverageInterpolation)(x::AbstractVector)
    m = length(itp.functionals)
    contributions = Vector{eltype(itp.c)}(undef, m)
    Threads.@threads for j in 1:m
        func = itp.functionals[j]
        contributions[j] = itp.c[j] *
            ustrip(integral(q -> itp.kernel(x, _to_coords(q)), func.volume;
                            # ibackend = Backend.Quadrature(gausslegendre(10))
                            )) /
            func.volume_measure
    end
    return sum(contributions)
end

function (itp::KernelInterpolation.CellAverageInterpolation{1})(x::Real)
    return itp([x])
end

# ── CellAverageFunctional evaluation ─────────────────────────────────────────

# λ(f) = (1/|V|) ∫_V f(x) dx  where x is a plain SVector{Dim,Float64}.
function (func::KernelInterpolation.CellAverageFunctional)(f)
    return ustrip(integral(p -> f(_to_coords(p)), func.volume)) / func.volume_measure
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

function KernelInterpolation.centroid_enclosing_radius(geom::Meshes.Geometry)
    c = centroid(geom)
    return ustrip.(maximum(norm(v - c) for v in vertices(geom)))
end

function KernelInterpolation.centroid_enclosing_radius(geoms::AbstractVector{<:Meshes.Geometry})
    return ustrip.(maximum(KernelInterpolation.centroid_enclosing_radius, geoms))
end

function KernelInterpolation.centroid_nodeset(
        functionals::Vector{<:KernelInterpolation.CellAverageFunctional})
    coords = [_to_coords(centroid(func.volume)) for func in functionals]
    return KernelInterpolation.NodeSet(coords)
end

# ── Fill distance ─────────────────────────────────────────────────────────────

@doc raw"""
    fill_distance(nodeset, domain::Meshes.Geometry; n_ref = 2000)

Approximate the fill distance
```math
    h_{X,\Omega} = \sup_{x \in \Omega} \min_{x_j \in X} \|x - x_j\|_2
```
by sampling `domain` with `n_ref` points drawn from
`Meshes.HomogeneousSampling` and forwarding to the two-argument
[`fill_distance(nodeset, reference)`](@ref).

!!! warning "Stochastic result"
    The reference points are drawn randomly. Different calls with the same
    arguments may return slightly different values. For reproducible results
    either call `Random.seed!` before this function, or build a deterministic
    reference [`NodeSet`](@ref) manually and use the two-argument form.

See also [`separation_distance`](@ref).
"""
function KernelInterpolation.fill_distance(nodeset::KernelInterpolation.NodeSet,
                                           domain::Meshes.Geometry;
                                           n_ref::Int = 2000)
    @warn "fill_distance: reference points are drawn randomly via HomogeneousSampling($n_ref). " *
          "The result is stochastic — repeated calls may differ slightly. " *
          "Call `Random.seed!` beforehand or pass a pre-built reference NodeSet " *
          "to the two-argument form for reproducible results." maxlog=1
    ref_pts   = Meshes.sample(domain, Meshes.HomogeneousSampling(n_ref))
    reference = KernelInterpolation.NodeSet([_to_coords(p) for p in ref_pts])
    return KernelInterpolation.fill_distance(nodeset, reference)
end

# ── Cell geometry constructors ────────────────────────────────────────────────

# Return N^dim non-overlapping boxes from a uniform RegularGrid on [a,b]^dim.
function KernelInterpolation.regular_cells(N::Int; a = 0.0, b = 1.0, dim::Int = 1)
    lo   = ntuple(_ -> Float64(a), dim)
    hi   = ntuple(_ -> Float64(b), dim)
    dims = ntuple(_ -> N, dim)
    return collect(elements(RegularGrid(lo, hi; dims)))
end

# Return N^dim + (N-1)^dim boxes: a primary RegularGrid plus a half-cell-shifted copy.
# The staggered layout places centroid nodes between primary cells, which improves kernel
# matrix conditioning compared to a single uniform grid of the same total density.
function KernelInterpolation.overlapping_cells(N::Int; a = 0.0, b = 1.0, dim::Int = 1)
    lo   = ntuple(_ -> Float64(a), dim)
    hi   = ntuple(_ -> Float64(b), dim)
    dims = ntuple(_ -> N, dim)
    primary  = collect(elements(RegularGrid(lo, hi; dims)))
    h        = (Float64(b) - Float64(a)) / N
    lo_shift = ntuple(_ -> Float64(a) + h / 2, dim)
    hi_shift = ntuple(_ -> Float64(b) - h / 2, dim)
    dims_s   = ntuple(_ -> N - 1, dim)
    secondary = collect(elements(RegularGrid(lo_shift, hi_shift; dims = dims_s)))
    return vcat(primary, secondary)
end

# Partition [a,b]² into 2N² right triangles by splitting each square cell along its
# lower-left to upper-right diagonal. Returned as a Vector{Triangle}.
function KernelInterpolation.triangular_cells(N::Int; a = 0.0, b = 1.0)
    h    = (Float64(b) - Float64(a)) / N
    tris = Vector{Triangle}(undef, 2 * N^2)
    k    = 0
    for j in 0:(N - 1), i in 0:(N - 1)
        p00 = Point(a + i * h,       a + j * h)
        p10 = Point(a + (i + 1) * h, a + j * h)
        p01 = Point(a + i * h,       a + (j + 1) * h)
        p11 = Point(a + (i + 1) * h, a + (j + 1) * h)
        tris[k += 1] = Triangle(p00, p10, p01)   # lower-left triangle (CCW)
        tris[k += 1] = Triangle(p10, p11, p01)   # upper-right triangle (CCW)
    end
    return tris
end

# Tessellate a point set and return the mesh cells as a plain Vector of geometries.
# Accepts a Meshes.PointSet, a KernelInterpolation.NodeSet, or any iterable of
# Meshes.Point / AbstractVector coordinates. Pass method = DelaunayTesselation() or
# VoronoiTesselation(); the result is a Vector{Triangle} or Vector{Ngon} respectively.
function KernelInterpolation.tessellation_cells(points, method)
    if points isa PointSet
        pset = points
    else
        pts_vec = [p isa Point ? p : Point(Float64.(p)...) for p in points]
        pset    = PointSet(pts_vec)
    end
    mesh = tesselate(pset, method)
    return [element(mesh, i) for i in 1:nelements(mesh)]
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
