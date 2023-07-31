import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_motion(time_signal_proximal, main_signal_proximal, time_signal_distal, main_signal_distal, title):
    """
    function to plot the motion of a signal - acceleration or gyroscope plots supported
    :param time_signal_proximal: list; list of time values in seconds
    :param main_signal_proximal: list; list of signal values
    :param time_signal_distal: list; list of time values in seconds
    :param main_signal_distal: list; list of signal values
    :param title: string; title of the plot
    :return: None
    """
    # Create a Plotly figure with two subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Proximal', 'Distal'))
    # Check if the signal is a 3D signal
    if isinstance(main_signal_proximal[0], (list, np.ndarray)):
        if isinstance(time_signal_proximal, np.ndarray): time_signal_proximal = time_signal_proximal.tolist()
        if isinstance(time_signal_distal, np.ndarray): time_signal_distal = time_signal_distal.tolist()

        # Add the proximal signal to the first subplot
        fig.add_trace(
            go.Scatter(x=time_signal_proximal, y=[val[0] for val in main_signal_proximal], mode='lines', name='X',
                       legendgroup=1),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=time_signal_proximal, y=[val[1] for val in main_signal_proximal], mode='lines', name='Y',
                       legendgroup=1),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(x=time_signal_proximal, y=[val[2] for val in main_signal_proximal], mode='lines', name='Z',
                       legendgroup=1),
            row=1, col=1)
        # Add the distal signal to the second subplot
        fig.add_trace(go.Scatter(x=time_signal_distal, y=[val[0] for val in main_signal_distal], mode='lines', name='X',
                                 legendgroup=2),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=time_signal_distal, y=[val[1] for val in main_signal_distal], mode='lines', name='Y',
                                 legendgroup=2),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=time_signal_distal, y=[val[2] for val in main_signal_distal], mode='lines', name='Z',
                                 legendgroup=2),
                      row=2, col=1)
    # Check if the signal is a 1D signal and motion data
    elif isinstance(main_signal_proximal[0], (np.float64, float)) and ('Gyro' in title or 'Acc' in title):
        # Add the proximal signal to the first subplot
        fig.add_trace(go.Scatter(x=time_signal_proximal, y=main_signal_proximal, mode='lines', name='RMS Proximal',
                                 legendgroup=1),
                      row=1, col=1)
        # Add the distal signal to the second subplot
        fig.add_trace(
            go.Scatter(x=time_signal_distal, y=main_signal_distal, mode='lines', name='RMS Distal', legendgroup=2),
            row=2, col=1)
        # Check if signal is a 1D signal and joint angles
    elif isinstance(main_signal_proximal[0], (np.float64, float)) and 'Angle' in title:
        # Add the proximal signal to the first subplot
        fig.add_trace(go.Scatter(x=time_signal_proximal, y=main_signal_proximal, mode='lines', name='Angle Proximal',
                                 legendgroup=1),
                      row=1, col=1)
        # Add the distal signal to the second subplot
        fig.add_trace(
            go.Scatter(x=time_signal_distal, y=main_signal_distal, mode='lines', name='Angle Distal', legendgroup=2),
            row=2, col=1)

    # Set title of the figure
    fig.update_layout(title_text=title)
    # Show the figure
    fig.show(show_plots_in_tool_window=False)


def plot_fft(time_signal, main_signal, title):
    """
    function to plot the fft of a signal
    :param time_signal: list; list of time values in seconds
    :param main_signal: list; list of signal values
    :param title: string; title of the plot
    :return:
    frequencies list; list of frequencies
    fft_result list; list of magnitude at positive frequencies
    """
    # Normalize the main signal to oscillate around zero
    mean_signal = np.mean(main_signal)
    normalized_signal = main_signal - mean_signal

    # Calculate the Fast Fourier Transform (FFT) of the normalized signal
    fft_result = np.fft.fft(normalized_signal)
    n = len(fft_result)
    timestep = time_signal[1] - time_signal[0]
    frequencies = np.fft.fftfreq(n, d=timestep)

    # Only consider non-negative frequencies (positive side of the spectrum)
    positive_freq_indices = frequencies >= 0
    frequencies = frequencies[positive_freq_indices]
    fft_result = fft_result[positive_freq_indices]

    # Identify the x and y values of the peak in the signal
    peak_index = np.argmax(np.abs(fft_result))
    peak_freq = frequencies[peak_index]
    peak_mag = np.abs(fft_result[peak_index])

    # Create a Plotly figure
    fig = go.Figure()

    # Plot the magnitude of the FFT result
    fig.add_trace(go.Scatter(x=frequencies, y=np.abs(fft_result),
                             mode='lines', name='FFT Magnitude'))
    # Add x on the plot for the peak frequency
    fig.add_trace(go.Scatter(x=[peak_freq], y=[peak_mag],
                                mode='markers', name='Peak Frequency'))

    # Add the frequency of peak to the title
    title += f' (Peak @ {peak_freq:.2f} Hz)'

    # Set plot labels and title
    fig.update_layout(title_text=title,
                      xaxis_title='Frequency (Hz)',
                      yaxis_title='Magnitude',
                      showlegend=True)

    # Show the plot
    fig.show()
    return frequencies, np.abs(fft_result)
