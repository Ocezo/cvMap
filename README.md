# cvData

`cvData` is a small C++/OpenCV project used to generate a handwritten character dataset from scanned sheets.

The goal is to:
- start from a sheet containing handwritten digits or letters;
- detect the grid;
- extract each cell;
- generate normalized and binarized image samples;
- write the matching labels file.

On the `figures` branch, the program works with handwritten digits (`0` to `9`) using the images stored in `img/in/figures/`.

The source sheets also use alternating columns, as in `0n1s.jpg`, so even and odd labels can be distinguished while scanning the grid from left to right and top to bottom.

Each sheet contains 140 labels arranged as 14 rows by 10 columns. With a maximum `scale_factor` of `10`, a single grid such as `0n1s.jpg` can therefore generate up to 1400 labels.

## Project contents

- `cvData.cpp`: main program for detection, extraction, and dataset generation.
- `CMakeLists.txt`: CMake build configuration.
- `img/in/`: input images (grid, digits, letters).
- `img/out/`: generated outputs (`rois`, binarized images, detected lines, Harris corners, labels).
- `tools/`: helper files for generating an empty grid.

## Build with CMake

Switch to the `figures` branch:

```bash
git checkout figures
```

Then build the project:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

## Run

From the `build` directory, run:

```bash
./cvData
```

The program reads images from `img/in/figures/` and writes results to `img/out/`.

## Requirements

- CMake >= 3.20
- a C++17-compatible compiler
- OpenCV
