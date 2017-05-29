#include "multiobject.h"
#include <cmath>
using namespace::hzl;

MultiObject::MultiObject()
{
    initData();
}

MultiObject::~MultiObject(){}

//初始化数据
void MultiObject::initData()
{
    int i,j;
    for(i=0; i<max_m; ++i)
    {
        mb[i].x = -1;
        mb[i].y = -1;
        cur_p[i] = 0;
        for(j=0; j<max_n; ++j)
        {
            wz[i][j].x = -1;
            wz[i][j].y = -1;
        }
        pos_freed.insert(max_m);
    }
    object_num = 0;
    cur_mb_end = 0;
}

bool MultiObject::isNewObject(int x,int y, int height, int width)
{
    int cur_ymin = ymin * height;
    int cur_ymax = ymax * height;
    int cur_xmin = ymin * width;
    int cur_xmax = ymax * width;

    if( y < cur_ymin || y > cur_ymax || x < cur_xmin || x > cur_xmax )
        return true;
    else
        return false;
}

/*bool MultiObject::isOutOfVision(int x,int y, int height, int width)
{
    int cur_ymin = ymin * height;
    int cur_ymax = ymax * height;
    int cur_xmin = ymin * width;
    int cur_xmax = ymax * width;

    if( y < cur_ymin || y > cur_ymax || x < cur_xmin || x > cur_xmax )
        return true;
    else
        return false;
}*/

void MultiObject::addNewObject(int x,int y)
{
    if(object_num >= max_m)
        return;
    else
    {
        mb[object_num].x = x;
        mb[object_num].y = y;
        object_num++;
    }
}
/*功能：将释放的数组元素下标加入一个链表中，便于以后分配。
输入：刚释放的数组元素的下标xh;
输出：无输出.*/
void MultiObject::Addkw(int xh)
{
    pos_freed.insert(xh);
}
/*功能：有新的目标出现在运动区域，找到合适的数组下标，将此数组空间分配给该目标。
输入：无输入，直接调用该函数，从kwhead所指的链表中（为空闲数组下标）依次分
配空间。
输出：输出要分配的数组下标。*/
int MultiObject::Delkw()
{
    int index;
    if(!pos_freed.empty())
    {
        for(int i=0; i<max_m; ++i)
        {
            if(pos_freed.count(i)==1)
            {
                index = i;
                pos_freed.erase(i);
                break;
            }
        }
        return index;
    }
    else
        return -1;
}

//输入目标框集合
void MultiObject::findObject(Point *object,size_t size)
{
    unsigned int i,j;
    unsigned int pos;
    double min_distance,temp_dis;

    if(size == 0)
        return ;
    if(pos_haved.empty()) //如果没有老的目标 直接加入旧目标
    {
        for(i=0; i<size ; ++i)
        {
            mb[i] = object[i];
            pos_haved.insert(i);
            pos_freed.erase(i);

            cur_p[i] = 0;   //定位到初始位置
            wz[i][cur_p[i]] = object[i];  //加入到序列当中
            cur_p[i]++;
        }
    }
    else
    {
        std::set<int> have_searched;
        for(i=0; i<size; ++i)  //遍历每个新的目标框
        {
            min_distance = INFINITY;
            for(j=0; j<max_m; ++j)
            {
                if(pos_haved.count(j)==0 && have_searched.count(j)==1)
                    continue;

                temp_dis =  sqrt(pow((mb[i].x-object[j].x),2)+ pow((mb[i].y-object[j].y),2));
                if(min_distance > temp_dis)
                {
                    min_distance = temp_dis;
                    pos = j;
                }
            }
            //找到最短的目标 如果满足最小距离阈值则接受
            if(min_distance < threshhold)
            {
                have_searched.insert(pos);
                //可以在这里加入适当的特征匹配来进行二次确认
                mb[pos] = object[i];

                wz[i][cur_p[i]] = object[i];
                cur_p[pos]++;
            }
            else //if(isNewObject(object[pos].x))   //认为是新的目标
            {
               int index = Delkw();
               pos_haved.insert(index);
               have_searched.insert(index);

               cur_p[index] = 0;   //定位到初始位置
               wz[i][cur_p[i]] = object[i];
               cur_p[index]++;

            }
        }
        for(i=0; i<max_m; ++i)
        {
            if(pos_haved.count(i)==1 && have_searched.count(i)==0)
            {
                //之前有记录现在没有记录了 先直接删除
               // if(isOutOfVision(mb[i].x, mb[i].y, )
                pos_haved.erase(i);
                pos_freed.insert(i);
            }
        }
    }
}
