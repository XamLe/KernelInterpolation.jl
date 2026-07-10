"""
    CellAverageFunctional

A functional that computes the cell average of a function over a control volume.

The functional is defined as:
    λ(f) = (1/|V|) ∫_V f(x) dx

where V is any parametrized geometry from Meshes.jl and |V| is its volume measure.
Integration is performed via `Meshes.integral` (h-adaptive cubature).

This functionality is provided by the KernelInterpolationCellAverageExt extension and
requires Meshes.jl to be loaded.

# Fields
- `volume::Any`: The control volume — any parametrized Meshes.jl geometry
- `volume_measure::RealT`: The volume measure |V| in the same precision as the geometry

The type parameter `RealT` is determined by the geometry's coordinate type (e.g. `Float64`
or `BigFloat`) and controls the floating-point precision used throughout quadrature.
"""
struct CellAverageFunctional{Dim, RealT <: Real}
    volume::Any          # any parametrized Meshes.jl geometry
    volume_measure::RealT
end

"""
    assemble_cell_average_matrix(functionals, kernel)

Assemble the kernel Gram matrix for cell-average interpolation. Entry ``(i,j)`` is

```math
    A_{ij} = \\frac{1}{|V_i||V_j|}\\int_{V_i}\\int_{V_j} K(x,y)\\,\\mathrm{d}y\\,\\mathrm{d}x.
```

The element type `RealT` of the returned matrix is inferred from the `RealT` parameter of
the `CellAverageFunctional`s, which is determined by the coordinate type of the underlying
geometry. Construct the functionals with BigFloat geometry (via `regular_cells(...; RealT=BigFloat)`)
and call `setprecision(BigFloat, bits)` beforehand to perform assembly and solve in higher
precision.
Requires Meshes.jl.

See also [`cell_average_interpolate`](@ref).
"""
function assemble_cell_average_matrix end

"""
    enclosing_radius(geometry; anchor = centroid(geometry))
    enclosing_radius(geometries; anchors = nothing)

Return the radius of the smallest ball centered at `anchor` that contains `geometry`,
i.e. the maximum distance from `anchor` to any vertex:
```math
    r(V, p) = \\max_{v \\in \\mathrm{vertices}(V)} \\|v - p\\|.
```
For a faceted geometry the maximum is always attained at a vertex, so the result is
exact. `anchor` defaults to `centroid(geometry)`.

For a vector of geometries, returns a `Vector{Float64}` of per-cell radii.
`anchors` may be a matching vector of anchor points (e.g. Voronoi seeds);
if omitted the centroid of each geometry is used.

Requires `Meshes.jl`. Works for any geometry with `vertices` defined.
"""
function enclosing_radius end

"""
    diameter(geometry)
    diameter(geometries)

Return the diameter of a faceted geometry, i.e. the maximum distance between
any two vertices:
```math
    \\mathrm{diam}(V) = \\max_{v, w \\in \\mathrm{vertices}(V)} \\|v - w\\|.
```
This is exact for any faceted geometry, convex or not: the diameter of a set equals
the diameter of its convex hull, which is always attained at a pair of extreme points
— a subset of the polygon's own vertices.
Requires `Meshes.jl`. Works for any geometry with `vertices` defined.
"""
function diameter end

"""
    centroid_nodeset(functionals)

Return a [`NodeSet`](@ref) containing the centroids of all control volumes in
`functionals`. Requires Meshes.jl.

See also [`separation_distance`](@ref), [`maximum_cell_diameter`](@ref).
"""
function centroid_nodeset end
