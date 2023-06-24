import pandas as pd
import numpy as np
import scipy.signal as signal
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table


def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    smoothed = np.convolve(data, window, mode='same')
    return smoothed


def fit_baseline(data, window_size, order):
    x = np.arange(len(data))
    coeffs = np.polyfit(x, data, order)
    baseline = np.polyval(coeffs, x)
    return baseline


def create_fig(data, time, voltage, baseline, baseline_removed, peaks, peak_heights, peak_widths, peak_areas):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time, y=voltage, name='Original', visible='legendonly'))
    fig.add_trace(go.Scatter(x=time, y=baseline, name='Baseline', visible='legendonly', showlegend=True))
    fig.add_trace(go.Scatter(x=time, y=baseline_removed, name='Baseline Removed'))
    fig.add_trace(go.Scatter(x=time[peaks], y=baseline_removed[peaks], mode='markers', name='Peaks',
                             hovertemplate="Time: %{customdata[0]:.2f} s<br>" +
                                           "Peak Height: %{customdata[1]:.6f} V<br>" +
                                           "FWHM: %{customdata[2]:.2f} s<br>" +
                                           "Peak Area: %{customdata[3]:.3f} V*s",
                             customdata=np.column_stack(
                                 (time[peaks], peak_heights, peak_widths * (time[1] - time[0]), peak_areas))))

    fig.update_layout(title='数据处理结果', xaxis_title='Time (s)', yaxis_title='Voltage (V)')
    fig.update_layout(legend=dict(x=0, y=1))

    return fig


def process_data(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 提取时间和电压列
    time = data.iloc[:, 0]
    voltage = data.iloc[:, 1]

    # 平滑处理
    smoothed_voltage = moving_average(voltage, window_size=3)

    # 计算电压的10%-90%范围内的平均值
    voltage_range = voltage.quantile([0.1, 0.9])
    voltage_mean = voltage[(voltage >= voltage_range[0.1]) & (voltage <= voltage_range[0.9])].mean()

    # 限幅电压范围
    clamped_voltage = voltage.clip(lower=0.85 * voltage_mean, upper=1.25 * voltage_mean)

    # 基线拟合 多项式拟合
    baseline = fit_baseline(clamped_voltage, window_size=len(clamped_voltage) // 500, order=11)

    # 基线移除
    baseline_removed = smoothed_voltage - baseline

    # 峰识别
    peaks, properties = signal.find_peaks(baseline_removed, height=0.0002, distance=1)

    # 计算峰高和峰面积
    peak_heights = baseline_removed[peaks]
    peak_widths = signal.peak_widths(baseline_removed, peaks)[0]

    peak_areas = np.zeros_like(peak_heights)  # 初始化峰面积数组

    for i in range(len(peaks)):
        peak_start = int(peaks[i] - peak_widths[i] // 2)
        peak_end = int(peaks[i] + peak_widths[i] // 2)
        peak_areas[i] = np.trapz(baseline_removed[peak_start:peak_end + 1], dx=time[1] - time[0])

    # 计算FWHM
    fwhm = peak_widths * (time[1] - time[0])

    # 过滤峰信息
    fwhm_threshold_lower = 0.3
    fwhm_threshold_upper = 6.0
    peak_area_threshold_lower = 0.005
    peak_area_threshold_upper = 10.0

    filtered_peaks = []
    for i in range(len(peaks)):
        if fwhm_threshold_lower <= fwhm[i] <= fwhm_threshold_upper and \
                peak_area_threshold_lower <= peak_areas[i] <= peak_area_threshold_upper:
            filtered_peaks.append(peaks[i])

    filtered_peak_heights = baseline_removed[filtered_peaks]
    filtered_peak_widths = signal.peak_widths(baseline_removed, filtered_peaks)[0]
    filtered_peak_areas = np.zeros_like(filtered_peak_heights)

    for i in range(len(filtered_peaks)):
        peak_start = int(filtered_peaks[i] - filtered_peak_widths[i] // 2)
        peak_end = int(filtered_peaks[i] + filtered_peak_widths[i] // 2)
        filtered_peak_areas[i] = np.trapz(baseline_removed[peak_start:peak_end + 1], dx=time[1] - time[0])

    # 创建图形对象
    fig = create_fig(data, time, voltage, baseline, baseline_removed, filtered_peaks, filtered_peak_heights,
                     filtered_peak_widths, filtered_peak_areas)

    # 创建峰信息表格
    peak_info = pd.DataFrame({
        'Index': np.arange(1, len(filtered_peaks) + 1),
        'Time (s)': time[filtered_peaks],
        'Peak Height (V)': np.round(filtered_peak_heights, decimals=6),
        'FWHM (s)': np.round(filtered_peak_widths * (time[1] - time[0]), decimals=2),
        'Peak Area (V*s)': np.round(filtered_peak_areas, decimals=3)
    })

    # 保存峰数据和峰信息
    data['Baseline'] = baseline
    data['Baseline Removed'] = baseline_removed
    data.to_csv("data_modify_time_handle.csv", index=False)
    peak_info.to_csv("data_modify_time_peaks_info.csv", index=False)

    return fig, peak_info


# 运行应用
fig, peak_info = process_data('raw_data.csv')

# 创建Dash应用
app = Dash(__name__)

# 布局
app.layout = html.Div([
    html.H1('数据处理结果'),
    dcc.Graph(figure=fig),
    html.H2('峰信息表格'),
    dash_table.DataTable(
        id='peak-info',
        columns=[{"name": col, "id": col} for col in peak_info.columns],
        data=peak_info.to_dict('records'),
        style_table={'width': '100%'},
        style_cell={'textAlign': 'left'},
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
