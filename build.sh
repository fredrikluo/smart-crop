clang++ smartcrop.cpp  `pkg-config --libs --cflags-only-I OpenEXR opencv` -framework Cocoa -lblas  --shared -o smartcrop.so
clang++  test.cpp  -L. smartcrop.so `pkg-config --libs --cflags-only-I OpenEXR opencv` -framework Cocoa -lpng -ljpeg -ltiff -lz -ljasper -o test
