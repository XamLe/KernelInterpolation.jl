module KernelInterpolationCellAverageExt

using LinearAlgebra: Symmetric, norm, cond
using Meshes: Meshes, Box, Segment, Quadrangle, Hexahedron, Point, Triangle, Polytope,
              measure, to, ustrip, integral, centroid, vertices, boundingbox, RegularGrid,
              elements, nelements, element, tesselate, DelaunayTesselation,
              VoronoiTesselation, PointSet, RowMaximum
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
    return Meshes.ustrip.(Meshes.to(p))
end

# ── CellAverageFunctional constructor ─────────────────────────────────────────

# RealT is inferred from the geometry's coordinate type via typeof(ustrip(measure(volume))).
# For Float64 geometry this gives Float64; for BigFloat geometry it gives BigFloat.
function KernelInterpolation.CellAverageFunctional(volume::Meshes.Geometry)
    vol_measure = ustrip(measure(volume))
    RealT = typeof(vol_measure)
    return KernelInterpolation.CellAverageFunctional{Meshes.embeddim(volume), RealT}(
        volume, vol_measure)
end

# ── Matrix assembly ───────────────────────────────────────────────────────────

# Approximates A[i,j] = (1/|Ωᵢ||Ωⱼ|) ∫_{Ωᵢ} ∫_{Ωⱼ} K(x,y) dy dx via nested
# h-adaptive integration. RealT is inferred from the functionals' type parameter so the
# entire pipeline (quadrature nodes, weights, kernel evaluations, accumulation) runs in
# that precision.
function _entry(func_i::KernelInterpolation.CellAverageFunctional,
                func_j::KernelInterpolation.CellAverageFunctional,
                kernel)
    inner(p) = ustrip(integral(q -> kernel(_to_coords(p), _to_coords(q)), func_j.volume))
    return ustrip(integral(p -> inner(p), func_i.volume)) /
           (func_i.volume_measure * func_j.volume_measure)
end

function KernelInterpolation.assemble_cell_average_matrix(
        functionals::AbstractVector{KernelInterpolation.CellAverageFunctional{Dim, RealT}},
        kernel::KernelInterpolation.AbstractKernel;
        n_gl::Union{Int, Nothing} = nothing) where {Dim, RealT}
    n         = length(functionals)
    A         = Matrix{RealT}(undef, n, n)
    n_entries = n * (n + 1) ÷ 2
    # BigFloat uses MPFR task-local state; sequential loop avoids pool contention.
    use_threads = !(RealT <: BigFloat)

    if isnothing(n_gl)
        @info "Assembling $(n)×$(n) cell-average matrix ($n_entries entries, h-adaptive, $(use_threads ? Threads.nthreads() : 1) thread(s), RealT=$RealT)"
        entry = (i, j) -> _entry(functionals[i], functionals[j], kernel)
    else
        all(f -> _is_box_like(f.volume), functionals) ||
            error("GL assembly requires Box-like geometries; got $(typeof(functionals[1].volume))")
        t1d, w1d = FastGaussQuadrature.gausslegendre(n_gl)
        t1d = RealT.(t1d)
        w1d = RealT.(w1d)
        cells_nw = [_gl_cell_nodes_weights(f, t1d, w1d, Val(Dim)) for f in functionals]
        @info "Assembling $(n)×$(n) cell-average matrix ($n_entries entries, GL n_gl=$n_gl, $(use_threads ? Threads.nthreads() : 1) thread(s), RealT=$RealT)"
        entry = (i, j) -> _gl_entry(cells_nw[i][1], cells_nw[i][2],
                                     cells_nw[j][1], cells_nw[j][2], kernel)
    end

    # Exploit kernel symmetry K(x,y) = K(y,x): compute upper triangle only.
    t = @elapsed if use_threads
        Threads.@threads for i in 1:n
            for j in i:n
                A[i, j] = entry(i, j)
                A[j, i] = A[i, j]
            end
        end
    else
        for i in 1:n
            for j in i:n
                A[i, j] = entry(i, j)
                A[j, i] = A[i, j]
            end
        end
    end
    κ = cond(Float64.(A))
    @info "Matrix assembly complete  ($(round(t; digits = 1))s)  cond(A) ≈ $(round(κ; sigdigits = 4))"
    return A
end

function KernelInterpolation.cell_average_interpolate(
    functionals::AbstractVector{KernelInterpolation.CellAverageFunctional{Dim, RealT}},
    values::AbstractVector,
    kernel::KernelInterpolation.AbstractKernel;
    n_gl = nothing,
    system_matrix = nothing,
    linsolve = nothing) where {Dim, RealT}
    n = length(functionals)
    @assert length(values) == n "number of values must match number of functionals"
    A = if isnothing(system_matrix)
        KernelInterpolation.assemble_cell_average_matrix(functionals, kernel; n_gl)
    else
        Matrix{RealT}(system_matrix)
    end
    # Wrap as Symmetric; kernel Gram matrices are SPD. Symmetric picks the upper triangle
    # as authoritative when independent adaptive evaluations of A[i,j]/A[j,i] differ.
    A_sym = Symmetric(A)
    t_solve = @elapsed c = KernelInterpolation.solve_linear_system(A_sym, RealT.(values), linsolve)
    @info "Linear solve complete  ($(round(t_solve; digits = 3))s)"
    return KernelInterpolation.CellAverageInterpolation(kernel, collect(functionals), c, A_sym)
end

# ── Interpolant evaluation ────────────────────────────────────────────────────

# s(x) = Σⱼ cⱼ ψⱼ(x)  where  ψⱼ(x) = (1/|Ωⱼ|) ∫_{Ωⱼ} K(x,y) dy.
function (itp::KernelInterpolation.CellAverageInterpolation)(x::AbstractVector)
    use_threads = !(eltype(itp.c) <: BigFloat)
    contribution(j) = itp.c[j] *
        ustrip(integral(q -> itp.kernel(x, _to_coords(q)), itp.functionals[j].volume)) /
        itp.functionals[j].volume_measure
    contributions = similar(itp.c)
    if use_threads
        Threads.@threads for j in eachindex(contributions)
            contributions[j] = contribution(j)
        end
    else
        map!(contribution, contributions, eachindex(contributions))
    end
    return sum(contributions)
end

function (itp::KernelInterpolation.CellAverageInterpolation{1})(x::Real)
    return itp([x])
end

# ── GL expansion ─────────────────────────────────────────────────────────────

# Return (min_coords, max_coords) as plain RealT vectors for supported cell types.
# Box: direct min/max access (no vertices method on Box).
_cell_bounds(box::Meshes.Box, ::Type{RealT}) where {RealT} =
    RealT.(ustrip.(to(Meshes.minimum(box)))), RealT.(ustrip.(to(Meshes.maximum(box))))

# Segment, Quadrangle, Hexahedron: axis-aligned, so boundingbox gives exact bounds.
_cell_bounds(geom::Meshes.Geometry, ::Type{RealT}) where {RealT} =
    _cell_bounds(boundingbox(geom), RealT)

_is_box_like(geom) = geom isa Meshes.Box || geom isa Meshes.Segment ||
                     geom isa Meshes.Quadrangle || geom isa Meshes.Hexahedron

# Compute GL nodes (Dim × n_gl^Dim) and base weights w_jk/|V_j| for one cell.
# Shared by expand() and GL matrix assembly.
function _gl_cell_nodes_weights(func,
                                 t1d::AbstractVector{RealT},
                                 w1d::AbstractVector{RealT},
                                 ::Val{Dim}) where {RealT, Dim}
    mn, mx = _cell_bounds(func.volume, RealT)
    mid    = (mn .+ mx) ./ 2
    half   = (mx .- mn) ./ 2
    n_gl   = length(t1d)
    n      = n_gl^Dim
    nodes    = Matrix{RealT}(undef, Dim, n)
    bweights = Vector{RealT}(undef, n)
    for (k, idx) in enumerate(Iterators.product(ntuple(_ -> 1:n_gl, Val(Dim))...))
        for i in 1:Dim
            nodes[i, k] = mid[i] + half[i] * t1d[idx[i]]
        end
        bweights[k] = prod(w1d[idx[i]] * half[i] for i in 1:Dim) / func.volume_measure
    end
    return nodes, bweights
end

# A_{ij} ≈ Σ_k Σ_l bw_i[k] · K(y_ik, y_jl) · bw_j[l]
function _gl_entry(nodes_i, bw_i, nodes_j, bw_j, kernel)
    s = zero(promote_type(eltype(bw_i), eltype(bw_j)))
    for l in axes(nodes_j, 2)
        yj = view(nodes_j, :, l)
        for k in axes(nodes_i, 2)
            s += bw_i[k] * kernel(view(nodes_i, :, k), yj) * bw_j[l]
        end
    end
    return s
end

function KernelInterpolation.expand(
        itp::KernelInterpolation.CellAverageInterpolation{Dim, RealT},
        n_gl::Int) where {Dim, RealT}
    all(f -> _is_box_like(f.volume), itp.functionals) ||
        error("expand currently only supports Box/Segment/Quadrangle/Hexahedron geometries; got $(typeof(itp.functionals[1].volume))")

    t1d, w1d = FastGaussQuadrature.gausslegendre(n_gl)
    t1d = RealT.(t1d)
    w1d = RealT.(w1d)

    N          = length(itp.functionals)
    n_per_cell = n_gl^Dim
    nodes      = Matrix{RealT}(undef, Dim, N * n_per_cell)
    coeffs     = Vector{RealT}(undef, N * n_per_cell)

    for (j, func) in enumerate(itp.functionals)
        cell_nodes, bweights = _gl_cell_nodes_weights(func, t1d, w1d, Val(Dim))
        offset = (j - 1) * n_per_cell
        nodes[:, offset+1:offset+n_per_cell]  .= cell_nodes
        coeffs[offset+1:offset+n_per_cell]    .= itp.c[j] .* bweights
    end

    return KernelInterpolation.ExpandedCellAverageInterpolation{Dim, RealT,
                                                                 typeof(itp.kernel)}(
        itp.kernel, nodes, coeffs)
end

function (eitp::KernelInterpolation.ExpandedCellAverageInterpolation)(x::AbstractVector)
    s = zero(eltype(eitp.coefficients))
    for k in axes(eitp.nodes, 2)
        s += eitp.coefficients[k] * eitp.kernel(x, view(eitp.nodes, :, k))
    end
    return s
end

function (eitp::KernelInterpolation.ExpandedCellAverageInterpolation{1})(x::Real)
    return eitp([x])
end

# ── CellAverageFunctional evaluation ─────────────────────────────────────────

# λ(f) = (1/|V|) ∫_V f(x) dx  where x is a plain SVector{Dim,RealT}.
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

function KernelInterpolation.enclosing_radius(geom::Meshes.Geometry;
                                               anchor = centroid(geom))
    return ustrip(maximum(norm(v - anchor) for v in vertices(geom)))
end

function KernelInterpolation.enclosing_radius(geoms::AbstractVector{<:Meshes.Geometry};
                                               anchors = nothing)
    if isnothing(anchors)
        return [KernelInterpolation.enclosing_radius(g) for g in geoms]
    else
        return [KernelInterpolation.enclosing_radius(g; anchor = a)
                for (g, a) in zip(geoms, anchors)]
    end
end

function KernelInterpolation.diameter(geom::Meshes.Geometry)
    vs = vertices(geom)
    return ustrip(maximum(norm(vs[i] - vs[j]) for i in eachindex(vs) for j in (i+1):lastindex(vs)))
end

function KernelInterpolation.diameter(box::Meshes.Box)
    return ustrip(norm(Meshes.maximum(box) - Meshes.minimum(box)))
end

function KernelInterpolation.enclosing_radius(box::Meshes.Box;
                                               anchor = centroid(box))
    mn = to(Meshes.minimum(box))
    mx = to(Meshes.maximum(box))
    anc = to(anchor)
    # Distance from anchor is convex, so maximum over box is at a corner.
    # The maximizing corner picks min or max independently per dimension.
    return ustrip(sqrt(sum(max(abs(mn[i] - anc[i]), abs(mx[i] - anc[i]))^2
                           for i in 1:length(mn))))
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
    lo   = ntuple(_ -> a, dim)
    hi   = ntuple(_ -> b, dim)
    dims = ntuple(_ -> N, dim)
    return collect(elements(RegularGrid(lo, hi; dims)))
end

# Return (2N-1)^dim boxes of uniform width w = width_fraction*(b-a)/N with uniform
# spacing s = (b-a-w)/(2N-2), covering [a,b]^dim without gaps.  Every interior cell has
# an exclusive region of width 2s-w > 0, which improves kernel matrix conditioning over a
# pure tiling.  Requires width_fraction ∈ (1/2, 1); default 3/4.
function KernelInterpolation.overlapping_cells(N::Int; a = 0.0, b = 1.0, dim::Int = 1,
                                               width_fraction = 3 // 4)
    h = (b - a) / N
    w = width_fraction * h
    M = 2N - 1
    M == 1 && return [Box(ntuple(_ -> a, dim), ntuple(_ -> b, dim))]
    s      = (b - a - w) / (M - 1)
    starts = [a + k * s for k in 0:(M - 1)]
    return vec([Box(ntuple(d -> starts[I[d]], dim),
                    ntuple(d -> starts[I[d]] + w, dim))
                for I in Iterators.product(ntuple(_ -> 1:M, dim)...)])
end

# Partition [a,b]² into 2N² right triangles by splitting each square cell along its
# lower-left to upper-right diagonal. Returned as a Vector{Triangle}.
function KernelInterpolation.triangular_cells(N::Int; a = 0.0, b = 1.0)
    h    = (b - a) / N
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

function _bounds(func::KernelInterpolation.CellAverageFunctional)
    bb = boundingbox(func.volume)
    return _to_coords(minimum(bb)), _to_coords(maximum(bb))
end

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
