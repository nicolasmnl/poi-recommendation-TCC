-- CREATE MATERIALIZED VIEW pois_information AS 
-- SELECT poi.id, poi.category 
-- FROM poi
-- WHERE category.id = has_category.fk_category_id AND
-- poi.id = has_category.fk_poi_id;

-- -- SELECT checkin_count, name FROM pois_information WHERE id = 'mjNCnP7H2bCDKb3TuQCdgg';

-- CREATE MATERIALIZED VIEW bins_information AS
-- SELECT has_context_poi.fk_poi_id_center, has_context_poi.fk_poi_id_context, pois_information.checkin_count, 
-- pois_information.name, has_context_poi.distance_m, has_context_poi.fk_bin_number
-- FROM has_context_poi, pois_information
-- WHERE has_context_poi.fk_poi_id_context = pois_information.id;


-- CREATE MATERIALIZED VIEW bins_pois_information AS 
-- SELECT
-- bin.fk_poi_id_center,
-- pois_information.id as fk_poi_id_context,
-- pois_information.name,
-- pois_information.checkin_count,
-- pois_information.level,
-- ST_DistanceSphere(poi.geom, ST_INTERSECTION(bin.geom, poi.geom)) As distance_m,
-- bin.number
-- FROM poi, pois_information, bin
-- WHERE ST_INTERSECTS(bin.geom, poi.geom) = TRUE AND
-- poi.id = pois_information.id AND
-- bin.number < 5;


-- SELECT * FROM bins_information WHERE fk_poi_id_center = 'xwSz8ermIbzDbDH-1l3G4A' AND fk_bin_number = 13;

-- SELECT fk_poi_id_context, name, checkin_count, distance_m 
-- FROM bins_information 
-- WHERE fk_poi_id_center = 'mjNCnP7H2bCDKb3TuQCdgg' AND fk_bin_number = 26;

CREATE MATERIALIZED VIEW points_information AS
SELECT
osm_id,
access,
aeroway,
amenity,
barrier,
bicycle,
building,
covered,
foot,
highway,
historic,
horse,
intermittent,
junction,
landuse,
leisure,
man_made,
--military
motorcar,
planet_osm_point.natural,
-- power,
public_transport,
railway,
sport,
surface,
tourism,
tunnel,
water,
waterway,
z_order,
way
FROM planet_osm_point
WHERE
access is not null or
aeroway is not null or
amenity in (
'bicycle_parking',
'biergarten',
'bus_station',
'charging_station',
'community garden',
'clock',
'Condominium complex',
'coworking_space',
'drainage',
'Drainage',
'drinking_water',
'Flag',
'fire_station',
'fountain',
'fuel',
'grave_yard',
'kindergarten',
'letter_box',
'loading_dock',
'motorcycle_parking',
'parking',
'parking_space',
'police',
'post_box',
'post_office',
'prison',
'public_building',
'public_hall',
'ranger_station',
'recycling',
'sanitary_dump_station',
'shower',
'sporting goods',
'swimming_pool',
'taxi',
'telephone',
'toilets',
'townhall',
'trailer park',
'trailer_park',
'waste_basket',
'waste_disposal',
'whirlpool'
) or
barrier is not null or
bicycle is not null or
building is not null or
covered is not null or
foot is not null or
highway is not null or
historic is not null or
horse is not null or
intermittent is not null or
junction is not null or
landuse is not null or
leisure is not null or
man_made is not null or
--military
motorcar is not null or
planet_osm_point.natural is not null or
--power is not null or
public_transport is not null or
railway is not null or
sport is not null or
surface is not null or
tourism in (
'park',
'picnic_site',
'camp_pitc',
'theme_park',
'viewpoint'
) or
tunnel is not null or
water is not null or
waterway is not null;

CREATE MATERIALIZED VIEW roads_information AS
SELECT
osm_id,
access,
bicycle,
bridge,
covered,
cutting,
embankment,
foot,
highway,
historic,
horse,
junction,
oneway,
public_transport,
railway,
route,
service,
surface,
toll,
tunnel,
waterway,
z_order,
way
FROM planet_osm_roads
WHERE
access is not null or
bicycle is not null or
bridge is not null or 
covered is not null or
cutting is not null or
embankment is not null or
foot is not null or
highway is not null or
historic is not null or
horse is not null or
junction is not null or
oneway is not null or
public_transport is not null or
railway is not null or
route is not null or
service in (
'alley',
'parking_aisle',
'emergency_access',
'driveway'
) or
surface is not null or
toll is not null or
tunnel is not null or
waterway is not null;

CREATE MATERIALIZED VIEW polygons_information AS
SELECT
osm_id,
access,
aeroway,
amenity,
barrier,
bicycle,
boundary,
bridge,
building,
construction,
covered,
foot,
highway,
historic,
horse,
intermittent,
landuse,
leisure,
man_made,
--military
planet_osm_polygon.natural,
oneway,
--power,
public_transport,
railway,
service,
sport,
surface,
tourism,
water,
waterway,
wetland,
z_order,
way_area * 0.3048 ^ 2 way_area_m,
way
FROM planet_osm_polygon
WHERE
access is not null or
aeroway is not null or
amenity in (
'bicycle_parking',
'biergarten',
'bus_station',
'charging_station',
'community garden',
'clock',
'Condominium complex',
'coworking_space',
'drainage',
'Drainage',
'drinking_water',
'Flag',
'fire_station',
'fountain',
'fuel',
'grave_yard',
'kindergarten',
'letter_box',
'loading_dock',
'motorcycle_parking',
'parking',
'parking_space',
'police',
'post_box',
'post_office',
'prison',
'public_building',
'public_hall',
'ranger_station',
'recycling',
'sanitary_dump_station',
'shower',
'sporting goods',
'swimming_pool',
'taxi',
'telephone',
'toilets',
'townhall',
'trailer park',
'trailer_park',
'waste_basket',
'waste_disposal',
'whirlpool'
) or
barrier is not null or
bicycle is not null or
boundary is not null or
bridge is not null or 
(building is not null and building != 'yes') or
construction is not null or
covered is not null or
foot is not null or
highway is not null or
historic is not null or
horse is not null or
intermittent is not null or
landuse is not null or
leisure is not null or
man_made is not null or
--military
planet_osm_polygon.natural is not null or
oneway is not null or
--power is not null or
public_transport is not null or
railway is not null or
service is not null or
sport is not null or
surface is not null or
tourism in (
'theme_park',
'viewpoint',
'camp_site'
) or
water is not null or
waterway is not null or
wetland is not null;

CREATE MATERIALIZED VIEW polygons_building_information AS
SELECT
osm_id,
building,
way_area * 0.3048 ^ 2 way_area_m,
way
FROM planet_osm_polygon
WHERE
building = 'yes' AND
access is null and
aeroway is null and
amenity is null and
barrier is null and
bicycle is null and
bridge is null and
construction is null and
covered is null and
foot is null and
highway is null and
historic is null and
horse is null and
intermittent is null and
landuse is null and
leisure is null and
man_made is null and
planet_osm_polygon.natural is null and
oneway is null and
--power is not null or
public_transport is null and
railway is null and
service is null and
sport is null and
surface is null and
tourism is null and
water is null and
waterway is null and
wetland is null;



CREATE MATERIALIZED VIEW lines_information AS
SELECT
osm_id,
access,
aerialway,
aeroway,
barrier,
bicycle,
bridge,
construction,
covered,
cutting,
disused,
embankment,
foot,
highway,
historic,
horse,
intermittent,
junction,
landuse,
leisure,
man_made,
planet_osm_line.natural,
oneway,
public_transport,
railway,
route,
surface,
toll,
tourism,
--tracktype,
tunnel,
waterway,
z_order,
way
FROM planet_osm_line
WHERE
access is not null or
aerialway is not null or
aeroway is not null or
barrier is not null or
bicycle is not null or
bridge is not null or
construction is not null or
covered is not null or
cutting is not null or
disused is not null or
embankment is not null or
foot is not null or
highway is not null or
historic is not null or
horse is not null or
intermittent is not null or
junction is not null or
landuse is not null or
leisure is not null or
man_made is not null or
planet_osm_line.natural is not null or
oneway is not null or
public_transport is not null or
railway is not null or
route is not null or
surface is not null or
toll is not null or
tourism is not null or
--tracktype is not null or
tunnel is not null or
waterway is not null;



CREATE MATERIALIZED VIEW pois_roads_information AS 
SELECT
poi.id,  
roads_information.osm_id,
roads_information.access,
roads_information.bicycle,
roads_information.bridge,
roads_information.covered,
roads_information.cutting,
roads_information.embankment,
roads_information.foot,
roads_information.highway,
roads_information.historic,
roads_information.horse,
roads_information.junction,
roads_information.oneway,
roads_information.public_transport,
roads_information.railway,
roads_information.route,
roads_information.service,
roads_information.surface,
roads_information.toll,
roads_information.tunnel,
roads_information.waterway,
--ST_INTERSECTION(bin.geom, roads_information.way) as way_intersection
ST_Length(ST_Transform(ST_INTERSECTION(poi.context, roads_information.way), 3857)) AS length,
ST_DistanceSphere(poi.geom, ST_INTERSECTION(poi.context, roads_information.way)) As distance_m
FROM poi, roads_information
WHERE ST_INTERSECTS(poi.context, roads_information.way) = TRUE;
-- AND
-- bin.number < 11 AND
-- bin.fk_poi_id_center = poi.id;


CREATE MATERIALIZED VIEW pois_points_information AS 
SELECT 
poi.id, 
points_information.osm_id,
points_information.access,
points_information.aeroway,
points_information.amenity,
points_information.barrier,
points_information.bicycle,
points_information.building,
points_information.covered,
points_information.foot,
points_information.highway,
points_information.historic,
points_information.horse,
points_information.intermittent,
points_information.junction,
points_information.landuse,
points_information.leisure,
points_information.man_made,
--military
points_information.motorcar,
points_information.natural,
-- power,
points_information.public_transport,
points_information.railway,
points_information.sport,
points_information.surface,
points_information.tourism,
points_information.tunnel,
points_information.water,
points_information.waterway, 
ST_DistanceSphere(poi.geom, ST_INTERSECTION(poi.context, points_information.way)) As distance_m
FROM poi, points_information
WHERE ST_INTERSECTS(poi.context, points_information.way) = TRUE;



CREATE MATERIALIZED VIEW pois_polygons_information AS 
SELECT poi.id, 
polygons_information.osm_id,
polygons_information.access,
polygons_information.aeroway,
polygons_information.amenity,
polygons_information.barrier,
polygons_information.bicycle,
polygons_information.bridge,
polygons_information.building,
polygons_information.construction,
polygons_information.covered,
polygons_information.foot,
polygons_information.highway,
polygons_information.historic,
polygons_information.horse,
polygons_information.intermittent,
polygons_information.landuse,
polygons_information.leisure,
polygons_information.man_made,
polygons_information.natural,
polygons_information.oneway,
polygons_information.public_transport,
polygons_information.railway,
polygons_information.sport,
polygons_information.surface,
polygons_information.water,
polygons_information.waterway,
polygons_information.wetland,
polygons_information.way_area_m,
--ST_AREA(ST_Transform(ST_INTERSECTION(bin.geom, polygons_information.way), 3857)) * 0.3048 ^ 2 way_partial_area_m,
ST_DistanceSphere(poi.geom, ST_INTERSECTION(poi.context, polygons_information.way)) As distance_m
--ST_INTERSECTION(poi.context, polygons_information.way) as way_intersection
FROM poi, polygons_information
WHERE ST_INTERSECTS(poi.context, polygons_information.way) = TRUE;




CREATE MATERIALIZED VIEW pois_polygons_building_information AS 
SELECT 
poi.id,
count(polygons_building_information.osm_id) as building_count,
avg(ST_DistanceSphere(poi.geom, ST_INTERSECTION(poi.context, polygons_building_information.way))) as building_avg_distance,
SUM(
ST_AREA(ST_Transform(ST_INTERSECTION(poi.context, polygons_building_information.way), 3857)) * 0.3048 ^ 2) as area_total
FROM poi, polygons_building_information
WHERE ST_INTERSECTS(poi.context, polygons_building_information.way) = TRUE
GROUP BY poi.id;



CREATE MATERIALIZED VIEW pois_lines_information AS
SELECT
poi.id,
lines_information.osm_id,
lines_information.access,
lines_information.aerialway,
lines_information.aeroway,
lines_information.barrier,
lines_information.bicycle,
lines_information.bridge,
lines_information.construction,
lines_information.covered,
lines_information.cutting,
lines_information.disused,
lines_information.embankment,
lines_information.foot,
lines_information.highway,
lines_information.historic,
lines_information.horse,
lines_information.intermittent,
lines_information.junction,
lines_information.landuse,
lines_information.leisure,
lines_information.man_made,
lines_information.natural,
lines_information.oneway,
lines_information.public_transport,
lines_information.railway,
lines_information.route,
lines_information.surface,
lines_information.toll,
lines_information.tourism,
lines_information.tunnel,
lines_information.waterway,
ST_Length(ST_Transform(ST_INTERSECTION(poi.context, lines_information.way), 3857)) AS length,
ST_DistanceSphere(poi.geom, ST_INTERSECTION(poi.context, lines_information.way)) As distance_m
FROM poi, lines_information
WHERE ST_INTERSECTS(poi.context, lines_information.way) = TRUE;



-- Código para gerar a vizinhança de um POI utilizando as geometrias
SELECT bin.fk_poi_id_center,
pois_information.id AS fk_poi_id_context,
pois_information.name,
pois_information.checkin_count,
pois_information.level,
st_distancesphere(pcx.geom, pct.geom) AS distance_m,
bin.number AS fk_bin_number
FROM poi pcx,
poi pct,
pois_information,
bin
WHERE st_intersects(bin.geom, pcx.geom) = true AND 
pcx.id <> pct.id AND 
pcx.id = pois_information.idt AND 
pct.id = bin.fk_poi_id_center AND 
bin.number < 5;