project( mini-opencv )

cmake_minimum_required( VERSION 3.5 )

include_directories(
    ${CMAKE_HOME_DIRECTORY}/core
    ${CMAKE_HOME_DIRECTORY}/imgproc
    ${CMAKE_HOME_DIRECTORY}/include
    #/usr/local/include/opencv4
)

link_directories(
    ${CMAKE_HOME_DIRECTORY}/lib
)

# added the common source files
aux_source_directory(${CMAKE_HOME_DIRECTORY}/ PROJECT_SOURCE_FILES )
#aux_source_directory(${CMAKE_HOME_DIRECTORY}/core PROJECT_SOURCE_FILES )
#aux_source_directory(${CMAKE_HOME_DIRECTORY}/imgproc PROJECT_SOURCE_FILES )


# set the output library 
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/ )
set( PROJECT_OUTPUT_FILE_NAME app)
set( CMAKE_CXX_STANDARD 11)
	
add_executable( ${PROJECT_OUTPUT_FILE_NAME} ${PROJECT_SOURCE_FILES} )
#add_subdirectory(opencv)
target_link_libraries(
	${PROJECT_OUTPUT_FILE_NAME}
	opencv_calib3d 
	opencv_core 
	opencv_dnn 
	opencv_features2d 
	opencv_flann 
	opencv_highgui 
	opencv_imgcodecs 
	opencv_imgproc 
	opencv_ml
	opencv_objdetect
	opencv_photo
	opencv_stitching 
	opencv_video
	opencv_videoio
)


