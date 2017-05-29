#ifndef MULTIOBJECT_H
#define MULTIOBJECT_H

#include <set>
#include <cstddef>
namespace hzl
{

struct Point
{
    int x;
    int y;
};
const int max_m = 100;	//最多的目标数目
const int max_n = 100;	//目标最大的轨迹序列

const double ymax = 0.9;	//目标的质心阈值
const double ymin = 0.1;
const double threshhold = 0.2;

class MultiObject
{
public:
    Point wz[max_m][max_n];	//位置数组
    int cur_p[max_m];

    int object_num;		//当前目标的数量
    int cur_mb_end;
    Point mb[max_m];		//代表前一帧中所有运动目标链数组

    std::set<int> pos_freed;    //存放已经释放的数组
    std::set<int> pos_haved;

    MultiObject();
    ~MultiObject();

    bool isNewObject(int x,int y, int height, int width);
    void addNewObject(int x,int y);
    void initData();
    void findObject(Point *object,size_t size);
    int Delkw();
    void Addkw(int xh);
};
}
#endif // MULTIOBJECT_H
