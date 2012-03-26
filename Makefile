LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_flann -lopencv_features2d -lopencv_calib3d
CC = g++
all: main.o Homography.o
	$(CC) main.o Homography.o -o main $(LIBS)
main.o: main.cpp
	$(CC) -c main.cpp $(LIBS)
MeanShift.o: Homography.cpp Homography.h
	$(CC) -c Homography.cpp $(LIBS)
clean:
	rm *.o main
