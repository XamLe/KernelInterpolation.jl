"""
    CellAverageFunctional

A functional that computes the cell average of a function over a control volume.

The functional is defined as:
    λ(f) = (1/|V|) ∫_V f(x) dx

where V is a geometry from Meshes.jl and |V| is its volume measure.

This functionality is provided by the KernelInterpolationMeshesExt extension and requires Meshes.jl to be loaded.

# Fields
- `volume::Geometry`: The control volume (from Meshes.jl)
- `volume_measure::Float64`: The volume measure |V|
"""
struct CellAverageFunctional{Dim}
    volume::Any          # Geometry from Meshes.jl
    volume_measure::Float64
    quadrature::Any      # nothing, or (nodes, weights) precomputed by the extension
end

# Convenience constructor for cases without quadrature
CellAverageFunctional{Dim}(volume, measure::Float64) where {Dim} =
    CellAverageFunctional{Dim}(volume, measure, nothing)

function assemble_cell_average_matrix end

"""
    mesh_diameter(functionals)

Return the mesh diameter: the maximum cell diameter (length of the longest diagonal)
over all [`CellAverageFunctional`](@ref)s. Requires Meshes.jl.
"""
function mesh_diameter end

"""
    centroid_nodeset(functionals)

Return a [`NodeSet`](@ref) containing the centroids of all control volumes in
`functionals`. Requires Meshes.jl.

See also [`separation_distance`](@ref), [`mesh_diameter`](@ref).
"""
function centroid_nodeset end
