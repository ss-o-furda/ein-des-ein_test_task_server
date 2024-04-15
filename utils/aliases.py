from typing import TypeAlias


Vertex: TypeAlias = list[float]
Vertices: TypeAlias = list[Vertex]
Polygon: TypeAlias = dict[str, str | Vertices]
Polygons: TypeAlias = list[Polygon]
Classes: TypeAlias = list[str]
ResObject: TypeAlias = dict[str, str | Classes | Polygons]
