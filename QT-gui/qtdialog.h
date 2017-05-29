#ifndef QTDIALOG_H
#define QTDIALOG_H

#include <QDialog>
#include <opencv2/opencv.hpp>

namespace Ui {
class QTDialog;
}

class QTDialog : public QDialog
{
    Q_OBJECT

public:
    explicit QTDialog(QWidget *parent = 0);
    ~QTDialog();

private slots:
    void on_pushButton_1_clicked();

private:
    Ui::QTDialog *ui;

private:
    QString filepath;
    cv::VideoCapture video;
    cv::Mat myimage;
public:
    QImage MatToQImage(const cv::Mat& mat);
};

#endif // QTDIALOG_H
