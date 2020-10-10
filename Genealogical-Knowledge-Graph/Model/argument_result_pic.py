def draw_pic():
    fig = plt.figure(1, facecolor='white')  # 创建一个画布，背景为白色
    fig.clf()  # 画布清空
    # ax1是函数createPlot的一个属性，这个可以在函数里面定义也可以在函数定义后加入也可以
    ax = plt.subplot(111, frameon=True)  # frameon表示是否绘制坐标轴矩形


    res_mac_list = [86.46, 88.45, 90.50, 90.63, 91.30, 91.05, 91.34]
    res_mic_list = [89.38, 91.38, 92.37, 92.61, 93.09, 92.83, 92.70]
    augment_ratio_list = [0, 10, 20, 30, 40, 50, 60]

    plt.plot(augment_ratio_list, res_mic_list, marker=".", label="Module 1,2,3,4 (micro)", linestyle="-")
    plt.plot(augment_ratio_list, res_mac_list, marker="*", label="Module 1,2,3,4 (macro)", linestyle="--")

    res_12_mac_list = [82.03, 84.69, 88.19, 88.00, 89.14, 88.76, 89.56]
    res_12_mic_list = [93.43, 94.20, 95.23, 95.26, 95.55, 95.28, 95.21]
    plt.plot(augment_ratio_list, res_12_mic_list, marker=".", label="Module 1,2 (micro)", linestyle="-.")
    plt.plot(augment_ratio_list, res_12_mac_list, marker="*", label="Module 1,2 (macro)", linestyle=":")

    # plt.grid(True)
    plt.xticks([0, 10, 20, 30, 40, 50, 60])
    # plt.ylim((0.0, 5.0))
    plt.xlabel("Augment ratio (%)")
    plt.ylabel("F-1 measure (%)")
    plt.legend()


    plt.savefig("./figure_3.png", dpi=350)
    plt.show()


import matplotlib.pyplot as plt


if __name__=="__main__":
    data_dic = draw_pic()