#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
/*参数说明：data_im图像的数据
*         channels 图像的通道数
*         height 图像高度
*         width 图像宽度
*         ksize 卷积核的尺寸
*         stride 卷积核的步长
*         pad 边界填充的数量
*         data_col 输出的数据指针位置
* */
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1; //输出图像的大小
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    //最外层循环是每个卷积核的参数个数
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;   //计算对应于卷积核中的位置
        int h_offset = (c / ksize) % ksize; //卷积核的高度
        int c_im = c / ksize / ksize;   //哪一层卷积核
        //这两层循环是用卷积核把图像遍历一遍
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

