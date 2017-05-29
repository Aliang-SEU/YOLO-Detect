# Created by and for Qt Creator. This file was created for editing the project sources only.
# You may attempt to use it for building too, by modifying this file here.

#TARGET = QT-darknet

HEADERS = \
   $$PWD/src/activation_layer.h \
   $$PWD/src/activations.h \
   $$PWD/src/avgpool_layer.h \
   $$PWD/src/batchnorm_layer.h \
   $$PWD/src/blas.h \
   $$PWD/src/box.h \
   $$PWD/src/classifier.h \
   $$PWD/src/col2im.h \
   $$PWD/src/connected_layer.h \
   $$PWD/src/convolutional_layer.h \
   $$PWD/src/cost_layer.h \
   $$PWD/src/crnn_layer.h \
   $$PWD/src/crop_layer.h \
   $$PWD/src/cuda.h \
   $$PWD/src/data.h \
   $$PWD/src/deconvolutional_layer.h \
   $$PWD/src/demo.h \
   $$PWD/src/detection_layer.h \
   $$PWD/src/dropout_layer.h \
   $$PWD/src/gemm.h \
   $$PWD/src/gru_layer.h \
   $$PWD/src/im2col.h \
   $$PWD/src/image.h \
   $$PWD/src/layer.h \
   $$PWD/src/list.h \
   $$PWD/src/local_layer.h \
   $$PWD/src/matrix.h \
   $$PWD/src/maxpool_layer.h \
   $$PWD/src/network.h \
   $$PWD/src/normalization_layer.h \
   $$PWD/src/option_list.h \
   $$PWD/src/parser.h \
   $$PWD/src/region_layer.h \
   $$PWD/src/reorg_layer.h \
   $$PWD/src/rnn_layer.h \
   $$PWD/src/route_layer.h \
   $$PWD/src/shortcut_layer.h \
   $$PWD/src/softmax_layer.h \
   $$PWD/src/stb_image.h \
   $$PWD/src/stb_image_write.h \
   $$PWD/src/tree.h \
   $$PWD/src/utils.h \
    QT-gui/qtdialog.h \
    src/multiobject.h

SOURCES = \
   $$PWD/src/activation_layer.c \
   $$PWD/src/activations.c \
   $$PWD/src/art.c \
   $$PWD/src/avgpool_layer.c \
   $$PWD/src/batchnorm_layer.c \
   $$PWD/src/blas.c \
   $$PWD/src/box.c \
   $$PWD/src/captcha.c \
   $$PWD/src/cifar.c \
   $$PWD/src/classifier.c \
   $$PWD/src/coco.c \
   $$PWD/src/col2im.c \
   $$PWD/src/compare.c \
   $$PWD/src/connected_layer.c \
   $$PWD/src/convolutional_layer.c \
   $$PWD/src/cost_layer.c \
   $$PWD/src/crnn_layer.c \
   $$PWD/src/crop_layer.c \
   $$PWD/src/cuda.c \
   $$PWD/src/darknet.c \
   $$PWD/src/data.c \
   $$PWD/src/deconvolutional_layer.c \
   $$PWD/src/demo.c \
   $$PWD/src/detection_layer.c \
   $$PWD/src/detector.c \
   $$PWD/src/dice.c \
   $$PWD/src/dropout_layer.c \
   $$PWD/src/gemm.c \
   $$PWD/src/go.c \
   $$PWD/src/gru_layer.c \
   $$PWD/src/im2col.c \
   $$PWD/src/image.c \
   $$PWD/src/layer.c \
   $$PWD/src/list.c \
   $$PWD/src/local_layer.c \
   $$PWD/src/lsd.c \
   $$PWD/src/matrix.c \
   $$PWD/src/maxpool_layer.c \
   $$PWD/src/network.c \
   $$PWD/src/nightmare.c \
   $$PWD/src/normalization_layer.c \
   $$PWD/src/option_list.c \
   $$PWD/src/parser.c \
   $$PWD/src/region_layer.c \
   $$PWD/src/regressor.c \
   $$PWD/src/reorg_layer.c \
   $$PWD/src/rnn.c \
   $$PWD/src/rnn_layer.c \
   $$PWD/src/rnn_vid.c \
   $$PWD/src/route_layer.c \
   $$PWD/src/shortcut_layer.c \
   $$PWD/src/softmax_layer.c \
   $$PWD/src/super.c \
   $$PWD/src/swag.c \
   $$PWD/src/tag.c \
   $$PWD/src/tree.c \
   $$PWD/src/utils.c \
   $$PWD/src/voxel.c \
   $$PWD/src/writing.c \
   $$PWD/src/yolo.c \
    QT-gui/qtdialog.cpp \
    src/multiobject.cpp

INCLUDEPATH = \
    $$PWD/src

INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2

LIBS += /usr/local/lib/libopencv_calib3d.so \
/usr/local/lib/libopencv_core.so \
/usr/local/lib/libopencv_features2d.so \
/usr/local/lib/libopencv_flann.so \
/usr/local/lib/libopencv_highgui.so \
/usr/local/lib/libopencv_imgproc.so \
/usr/local/lib/libopencv_ml.so \
/usr/local/lib/libopencv_objdetect.so \
/usr/local/lib/libopencv_photo.so \
/usr/local/lib/libopencv_stitching.so \
/usr/local/lib/libopencv_superres.so \
/usr/local/lib/libopencv_video.so \
/usr/local/lib/libopencv_videostab.so

QT +=  core gui widgets
#DEFINES = 
INCLUDEPATH += /usr/local/include \
                /usr/local/include/opencv \
                /usr/local/include/opencv2
LIBS += /usr/local/lib/libopencv_highgui.so \
        /usr/local/lib/libopencv_core.so    \
        /usr/local/lib/libopencv_imgproc.so
FORMS += \
    QT-gui/qtdialog.ui

