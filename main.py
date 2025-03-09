#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import configparser
import random
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout,
    QGroupBox, QPushButton, QLineEdit, QLabel, QPlainTextEdit,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ---------------------- 训练线程 ----------------------
class TrainingWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)  # 新增的进度更新信号
    finished_signal = pyqtSignal(object, dict, dict)  # 返回 regressor, num_impute, cat_impute

    def __init__(self, selected_num_cols, selected_cat_cols, parent=None):
        super().__init__(parent)
        self.selected_num_cols = selected_num_cols
        self.selected_cat_cols = selected_cat_cols

    def run(self):
        import time
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from tabpfn import TabPFNRegressor

        # 设置随机种子
        RANDOM_SEED = 98
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.log_signal.emit("设置随机种子。")
        self.progress_signal.emit(10)

        # 读取 include.ini 文件（如果存在）
        config = configparser.ConfigParser()
        config.read("include.ini", encoding="utf-8")
        if "INCLUDE" in config and "features" in config["INCLUDE"]:
            include_features = [feat.strip() for feat in config["INCLUDE"]["features"].split(",")]
        else:
            ALL_NUM_COLS = ['Surface area', '孔径', '孔容', 'zeta', '-OH', '-CO3', '-COOH',
                             '-N', 'pH', 'Mo初始浓度', 'Dose', '吸附动力学']
            ALL_CAT_COLS = ['尺寸', '金属', '非金属']
            include_features = ALL_NUM_COLS + ALL_CAT_COLS

        self.log_signal.emit("选择的数值变量: " + str(self.selected_num_cols))
        self.log_signal.emit("选择的分类变量: " + str(self.selected_cat_cols))
        self.progress_signal.emit(20)

        DATA_PATH = "datas/Mo_baseline.csv"

        def load_and_preprocess_tabpfn(selected_num_cols, selected_cat_cols):
            """
            加载 CSV 数据、去除列首尾空格，
            将空字符串和 "/" 一律视为缺失值，
            数值型变量缺失值填充为 0，
            分类型变量填充为该列众数，
            返回特征 X 与目标 y。
            """
            df = pd.read_csv(DATA_PATH, na_values=['/'])
            df.columns = df.columns.str.strip()
            df.replace({"/": np.nan, "": np.nan}, inplace=True)
            # 数值变量处理：缺失值填充为 0
            if selected_num_cols:
                df_num = df[selected_num_cols].copy()
                for col in df_num.columns:
                    df_num[col] = df_num[col].fillna(0)
            else:
                df_num = pd.DataFrame(index=df.index)
            # 分类变量处理：缺失值填充为众数，并转换为字符串
            if selected_cat_cols:
                df_cat = df[selected_cat_cols].copy()
                for col in df_cat.columns:
                    df_cat[col] = df_cat[col].fillna(df_cat[col].mode()[0])
                    df_cat[col] = df_cat[col].astype(str)
            else:
                df_cat = pd.DataFrame(index=df.index)

            if not df_num.empty and not df_cat.empty:
                X = pd.concat([df_num, df_cat], axis=1)
            elif not df_num.empty:
                X = df_num
            elif not df_cat.empty:
                X = df_cat
            else:
                X = pd.DataFrame(index=df.index)

            y = df['Mo adsorption capacity']
            return X, y

        self.log_signal.emit("开始加载和预处理数据...")
        self.progress_signal.emit(30)
        try:
            X, y = load_and_preprocess_tabpfn(self.selected_num_cols, self.selected_cat_cols)
        except Exception as e:
            self.log_signal.emit("加载或预处理数据时出错: " + str(e))
            return
        self.progress_signal.emit(40)

        # 划分数据集（测试集比例 20%）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED)
        self.log_signal.emit(f"训练集样本数: {X_train.shape[0]}, 测试集样本数: {X_test.shape[0]}")
        self.progress_signal.emit(50)

        # 初始化并训练 TabPFN 模型
        regressor = TabPFNRegressor(model_path="tabpfn-v2-regressor.ckpt")
        self.log_signal.emit("开始训练 TabPFN 模型...")
        self.progress_signal.emit(60)
        try:
            regressor.fit(X_train, y_train)
        except Exception as e:
            self.log_signal.emit("训练模型时出错: " + str(e))
            return
        self.progress_signal.emit(80)

        # 在测试集上预测并评估模型
        predictions = regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions) * 100
        self.log_signal.emit(f"预测的均方误差 (MSE): {mse:.6f}")
        self.log_signal.emit(f"最终 R² 得分: {r2:.2f}%")
        self.progress_signal.emit(90)

        # 数值型变量缺失默认填充值统一设为 0
        num_impute = {col: 0 for col in self.selected_num_cols}
        # 分类型变量缺失填充值统一设为 "0"
        cat_impute = {col: "0" for col in self.selected_cat_cols}

        # 训练结束，传递模型和默认填充值
        self.finished_signal.emit(regressor, num_impute, cat_impute)
        self.progress_signal.emit(100)

# ---------------------- 主窗口 ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt 窗口化 TabPFN 应用")
        self.resize(800, 600)

        self.regressor = None
        self.num_impute = {}
        self.cat_impute = {}
        self.training_worker = None

        # 读取 include.ini 文件确定所选变量
        config = configparser.ConfigParser()
        config.read("include.ini", encoding="utf-8")
        ALL_NUM_COLS = ['Surface area', '孔径', '孔容', 'zeta', '-OH', '-CO3', '-COOH',
                          '-N', 'pH', 'Mo初始浓度', 'Dose', '吸附动力学']
        ALL_CAT_COLS = ['尺寸', '金属', '非金属']
        if "INCLUDE" in config and "features" in config["INCLUDE"]:
            include_features = [feat.strip() for feat in config["INCLUDE"]["features"].split(",")]
        else:
            include_features = ALL_NUM_COLS + ALL_CAT_COLS

        self.selected_num_cols = [feat for feat in ALL_NUM_COLS if feat in include_features]
        self.selected_cat_cols = [feat for feat in ALL_CAT_COLS if feat in include_features]

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # ---------------------- 模型训练模块 ----------------------
        training_group = QGroupBox("模型训练")
        training_layout = QVBoxLayout()
        training_group.setLayout(training_layout)

        self.train_button = QPushButton("训练模型")
        self.train_button.clicked.connect(self.on_train_clicked)
        training_layout.addWidget(self.train_button)

        # 日志输出框
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        training_layout.addWidget(self.log_text)

        # 新增训练进度条
        from PyQt5.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        training_layout.addWidget(self.progress_bar)

        main_layout.addWidget(training_group)

        # ---------------------- 模型推理模块 ----------------------
        inference_group = QGroupBox("模型推理")
        inference_layout = QVBoxLayout()
        inference_group.setLayout(inference_layout)

        # 创建表单输入，每个变量一个输入框
        form_layout = QFormLayout()
        self.input_fields = {}

        # 数值变量输入框
        for col in self.selected_num_cols:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("数值变量，留空或 '/' 使用默认值 0")
            self.input_fields[col] = line_edit
            form_layout.addRow(QLabel(col + "："), line_edit)

        # 分类型变量输入框
        for col in self.selected_cat_cols:
            line_edit = QLineEdit()
            line_edit.setPlaceholderText("分类变量，留空或 '/' 使用默认值 0")
            self.input_fields[col] = line_edit
            form_layout.addRow(QLabel(col + "："), line_edit)

        inference_layout.addLayout(form_layout)

        self.predict_button = QPushButton("预测")
        self.predict_button.clicked.connect(self.on_predict_clicked)
        inference_layout.addWidget(self.predict_button)

        self.result_label = QLabel("预测结果：")
        inference_layout.addWidget(self.result_label)

        main_layout.addWidget(inference_group)

    def log_message(self, message):
        """在日志框中追加日志信息"""
        self.log_text.appendPlainText(message)

    def on_train_clicked(self):
        """点击训练按钮，启动训练线程"""
        self.train_button.setEnabled(False)
        self.log_message("开始训练模型...")
        self.progress_bar.setValue(0)
        self.training_worker = TrainingWorker(self.selected_num_cols, self.selected_cat_cols)
        self.training_worker.log_signal.connect(self.log_message)
        self.training_worker.progress_signal.connect(self.progress_bar.setValue)
        self.training_worker.finished_signal.connect(self.on_training_finished)
        self.training_worker.start()

    def on_training_finished(self, regressor, num_impute, cat_impute):
        """训练完成回调，保存模型和默认填充值"""
        self.regressor = regressor
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.log_message("训练完成！")
        self.train_button.setEnabled(True)

    def on_predict_clicked(self):
        """点击预测按钮后采集输入，使用训练好的模型进行推理"""
        if not self.regressor:
            QMessageBox.warning(self, "警告", "尚未训练模型，请先训练模型！")
            return

        new_sample = {}
        # 处理数值变量输入
        for col in self.selected_num_cols:
            text = self.input_fields[col].text().strip()
            if text == "" or text == "/":
                new_sample[col] = 0
            else:
                try:
                    new_sample[col] = float(text)
                except ValueError:
                    self.log_message(f"输入错误，无法将『{text}』转换为数值，使用默认值 0。")
                    new_sample[col] = 0

        # 处理分类型变量输入
        for col in self.selected_cat_cols:
            text = self.input_fields[col].text().strip()
            if text == "" or text == "/":
                new_sample[col] = "0"
            else:
                new_sample[col] = text

        # 构造单样本 DataFrame（注意各特征顺序需与训练时一致）
        feature_order = self.selected_num_cols + self.selected_cat_cols
        sample_data = {col: [new_sample[col]] for col in feature_order}
        new_sample_df = pd.DataFrame(sample_data)

        self.log_message("输入的样本数据为：")
        self.log_message(str(new_sample_df))

        # 调用训练好的模型进行预测
        try:
            pred = self.regressor.predict(new_sample_df)
            self.result_label.setText("预测结果：" + str(pred[0]))
            self.log_message("预测结果：" + str(pred[0]))
        except Exception as e:
            self.log_message("预测时出错: " + str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
