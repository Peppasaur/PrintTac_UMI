// UMI gripper ArUco marker ID 1, DICT_4X4_50.
// Units: millimeters. The black outer edge is exactly 16 mm.

marker_edge = 16;
base_thickness = 1.20;
pattern_height = 0.15;
cell = marker_edge / 6;

// Top-view raster, row 0 is the top edge of the marker.
marker = [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1]
];

color("white") cube([marker_edge, marker_edge, base_thickness]);

for (row = [0:5]) {
    for (col = [0:5]) {
        if (marker[row][col] == 1) {
            color("black")
                translate([col * cell, (5 - row) * cell, base_thickness])
                    cube([cell, cell, pattern_height]);
        }
    }
}
