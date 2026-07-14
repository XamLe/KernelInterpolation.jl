@doc raw"""
    CellAverageInterpolation

Interpolation object for cell-average data. Represents the function
```math
    s(x) = \sum_{j=1}^N c_j\lambda_j(K(x, \cdot))
         = \sum_{j=1}^N \frac{c_j}{|V_j|}\int_{V_j} K(x, y)\,\mathrm{d}y,
```
where ``\lambda_j`` are [`CellAverageFunctional`](@ref)s and the coefficients ``c_j``
solve the linear system ``Ac = \bar{f}`` with matrix entries
```math
    A_{ij} = \frac{1}{|V_i||V_j|}\int_{V_i}\int_{V_j} K(x,y)\,\mathrm{d}y\,\mathrm{d}x.
```

Constructed via [`cell_average_interpolate`](@ref). Requires Meshes.jl.

# Fields
- `kernel`: the kernel function
- `functionals::Vector{CellAverageFunctional}`: the cell-average functionals
- `c::Vector{RealT}`: the coefficient vector
- `system_matrix`: the assembled (and possibly factorized) matrix ``A``

See also [`CellAverageFunctional`](@ref), [`cell_average_interpolate`](@ref).
"""
struct CellAverageInterpolation{Dim, RealT, KernelT, A}
    kernel::KernelT
    functionals::Vector{CellAverageFunctional{Dim, RealT}}
    c::Vector{RealT}
    system_matrix::A
end

function Base.show(io::IO, itp::CellAverageInterpolation)
    n = length(itp.functionals)
    return print(io,
                 "CellAverageInterpolation with $n functionals and kernel $(itp.kernel).")
end

"""
    interpolation_kernel(itp::CellAverageInterpolation)

Return the kernel of a [`CellAverageInterpolation`](@ref).
"""
interpolation_kernel(itp::CellAverageInterpolation) = itp.kernel

"""
    coefficients(itp::CellAverageInterpolation)

Return the coefficient vector of a [`CellAverageInterpolation`](@ref).
"""
coefficients(itp::CellAverageInterpolation) = itp.c

"""
    system_matrix(itp::CellAverageInterpolation)

Return the system matrix of a [`CellAverageInterpolation`](@ref).
"""
system_matrix(itp::CellAverageInterpolation) = itp.system_matrix

"""
    dim(itp::CellAverageInterpolation)

Return the spatial dimension of a [`CellAverageInterpolation`](@ref).
"""
dim(::CellAverageInterpolation{Dim}) where {Dim} = Dim

"""
    functionals(itp::CellAverageInterpolation)

Return the vector of [`CellAverageFunctional`](@ref)s of a
[`CellAverageInterpolation`](@ref).
"""
functionals(itp::CellAverageInterpolation) = itp.functionals

@doc raw"""
    cell_average_interpolate(functionals, values, kernel;
                             ibackend = nothing, dbackend = nothing, linsolve = nothing)

Interpolate cell-average `values` using the kernel `kernel` and the
[`CellAverageFunctional`](@ref)s in `functionals`. Determines the coefficients ``c_j``
in
```math
    s(x) = \sum_{j=1}^N c_j\lambda_j(K(x, \cdot))
```
by solving the linear system ``Ac = \bar{f}`` with matrix entries
```math
    A_{ij} = \frac{1}{|V_i||V_j|}\int_{V_i}\int_{V_j} K(x,y)\,\mathrm{d}y\,\mathrm{d}x.
```
Returns a [`CellAverageInterpolation`](@ref) that can be evaluated at any point.

Requires Meshes.jl. If `system_matrix` is provided it is used as the Gram matrix
directly and assembly is skipped — the caller is responsible for ensuring it was
assembled with the same `functionals` in the same order. If `linsolve` is provided
it is passed to LinearSolve.jl; otherwise the backslash operator is used.

See also [`CellAverageFunctional`](@ref), [`assemble_cell_average_matrix`](@ref).
"""
function cell_average_interpolate end

"""
    cell_averages(itp::CellAverageInterpolation)

Numerically compute the cell averages ``\\lambda_i(s)`` of the interpolant `itp`
over each of its control volumes. Requires Meshes.jl.
"""
function cell_averages end

"""
    ExpandedCellAverageInterpolation

A fast-evaluation form of [`CellAverageInterpolation`](@ref) obtained by pre-computing
Gauss-Legendre quadrature nodes and weights for each cell and collapsing the interpolant
into a plain nodal kernel sum. Evaluation at a point `x` costs `N * n_gl^Dim` kernel
evaluations with no adaptive quadrature overhead.

Constructed via [`expand`](@ref). Currently requires all control volumes to be
axis-aligned boxes (`Meshes.Segment` in 1D, `Meshes.Quadrangle` in 2D,
`Meshes.Hexahedron` in 3D, or `Meshes.Box`).

# Fields
- `kernel`: the kernel function
- `nodes::Matrix{RealT}`: GL nodes in physical space, stored column-wise (`Dim × (N·n_gl^Dim)`)
- `coefficients::Vector{RealT}`: combined coefficients ``\\tilde{c}_{jk} = c_j w_{jk} / |V_j|``
"""
struct ExpandedCellAverageInterpolation{Dim, RealT <: Real, KernelT}
    kernel::KernelT
    nodes::Matrix{RealT}
    coefficients::Vector{RealT}
end

function Base.show(io::IO, eitp::ExpandedCellAverageInterpolation{Dim}) where {Dim}
    n = length(eitp.coefficients)
    return print(io,
                 "ExpandedCellAverageInterpolation with $n GL nodes in $(Dim)D and kernel $(eitp.kernel).")
end

"""
    expand(itp::CellAverageInterpolation, n_gl::Int)

Pre-compute a `n_gl`-point-per-dimension Gauss-Legendre expansion of `itp`, returning an
[`ExpandedCellAverageInterpolation`](@ref) that evaluates without adaptive quadrature.

For each cell ``V_j`` the GL nodes ``y_{jk}`` in physical space and the combined
coefficients ``\\tilde{c}_{jk} = c_j w_{jk} / |V_j|`` are stored, so that

```math
    s(x) \\approx \\sum_{j,k} \\tilde{c}_{jk}\\, K(y_{jk}, x).
```

Currently requires all control volumes to be axis-aligned boxes (`Meshes.Segment` in 1D,
`Meshes.Quadrangle` in 2D, `Meshes.Hexahedron` in 3D, or `Meshes.Box`). Requires Meshes.jl.

See also [`CellAverageInterpolation`](@ref), [`ExpandedCellAverageInterpolation`](@ref).
"""
function expand end

"""
    regular_cells(N; a = 0.0, b = 1.0, dim = 1)

Return a `Vector` of `N^dim` non-overlapping axis-aligned boxes that uniformly tile
`[a, b]^dim`. Each box has side length `(b - a) / N`. Wraps `Meshes.RegularGrid`.
Requires Meshes.jl.

# Example
```julia
cells = regular_cells(8; a = 0.0, b = 1.0, dim = 1)   # 8 unit-interval cells
funcs = CellAverageFunctional.(cells)
```
"""
function regular_cells end

"""
    overlapping_cells(N; a = 0.0, b = 1.0, dim = 1, width_fraction = 3//4)

Return a `Vector` of `(2N-1)^dim` boxes of uniform width `w = width_fraction*(b-a)/N`,
placed with uniform spacing `s = (b-a-w)/(2N-2)` along each axis so that `[a,b]^dim` is
fully covered.  Every interior cell has an exclusive region of width `2s - w > 0`, which
improves kernel matrix conditioning over a single uniform tiling.  Requires
`width_fraction ∈ (1/2, 1)`. Requires Meshes.jl.

# Example
```julia
cells = overlapping_cells(8; a = 0.0, b = 1.0, dim = 1)   # (2*8-1) = 15 cells
funcs = CellAverageFunctional.(cells)
```
"""
function overlapping_cells end

"""
    triangular_cells(N; a = 0.0, b = 1.0)

Partition `[a, b]²` into `2N²` right triangles. The square domain is first divided into
an `N × N` regular grid; each square cell is then split along its lower-left to upper-right
diagonal into two CCW-oriented `Meshes.Triangle`s. Returns a `Vector{Meshes.Triangle}`.
Requires Meshes.jl.

# Example
```julia
tris  = triangular_cells(7; a = 0.0, b = 1.0)   # 98 triangles
funcs = CellAverageFunctional.(tris)
```
"""
function triangular_cells end

"""
    tessellation_cells(points, method)

Tessellate `points` using `method` (e.g. `Meshes.DelaunayTesselation()` or
`Meshes.VoronoiTesselation()`) and return the mesh cells as a plain `Vector` of
Meshes geometries (`Triangle`s for Delaunay, `Ngon`s for Voronoi). Accepts a
`Meshes.PointSet`, a `KernelInterpolation.NodeSet`, or any iterable whose elements
are either `Meshes.Point` objects or coordinate vectors / static arrays.
Requires Meshes.jl.

# Example
```julia
pts   = homogeneous_hypercube(20, -1.0, 1.0; dim = 2)
cells = tessellation_cells(pts, Meshes.VoronoiTesselation())   # Voronoi cells
funcs = CellAverageFunctional.(cells)
```
"""
function tessellation_cells end
