#include "qtdialog.h"
#include "ui_qtdialog.h"
#include <qfiledialog.h>

QTDialog::QTDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::QTDialog)
{
    ui->setupUi(this);
}

QTDialog::~QTDialog()
{
    delete ui;
}

/*
int main(int argc, char **argv)
{
    QApplication a(argc, argv);

    QTDialog *window = new QTDialog;
    window->setWindowTitle("车辆检测跟踪系统");

    window->show();
    return a.exec();
}*/

QImage QTDialog::MatToQImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        return QImage();
    }
}

void QTDialog::on_pushButton_1_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this, "open file", ".", "(Image File(*.png *.jpg *.jpeg *.bmp)");

    this->filepath = fileName;

    this->myimage = cv::imread(this->filepath.toStdString());
    cv::imshow("1",this->myimage);
    QImage img = MatToQImage(this->myimage);
    ui->label->setPixmap(QPixmap::fromImage(img));
    ui->label->resize(ui->label->pixmap()->size());
}
