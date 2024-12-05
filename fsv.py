import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

show_even = True
show_odd = True
use_components = False
component_lines = []
lockG = False

coef_C_0 = ...
coef_a = ...
coef_b = ...

def calculate_signal(N, f0, A):
    global t, show_even, show_odd, coef_C_0, coef_a, coef_b,use_components
    C0 = coef_C_0(A)

    if not show_even and not show_odd:
        K = 0
    elif show_even and show_odd:
        K = N
    elif show_even:
        K = N // 2
    elif show_odd:
        K = (N // 2) + (N % 2)

    a = np.zeros(K)
    b = np.zeros(K)
    theta = np.zeros(K)
    C = np.zeros(K)

    if show_even and show_odd:
        for n in range(1, K + 1):
            a[n - 1] = coef_a(A,n)
            b[n - 1] = coef_b(A,n)
    elif show_even:
        for n in range(1, K + 1):
            a[n - 1] = coef_a(A,2*n)
            b[n - 1] = coef_b(A,2*n)
    elif show_odd:
        for n in range(1, K + 1):
            a[n - 1] = coef_a(A,2*(n - 1)+1)
            b[n - 1] = coef_b(A,2*(n - 1)+1)

    for n in range(K):
        theta[n] = np.arctan2(-b[n], a[n])
        C[n] = np.sqrt(a[n]**2 + b[n]**2)
    if use_components:
        g = np.zeros((K+1,t.shape[0]))
        g[0] = C0*np.ones_like(t)
    else:
        g = C0 * np.cos(2 * np.pi * 0 * f0 * t)

    for n in range(1, K + 1):
        if show_even and show_odd:
            f_n = n * f0
        elif show_even:
            f_n = (2*n)*f0
        elif show_odd:
            f_n = (2*(n-1)+1)*f0
        if use_components:
            g[n] = C[n - 1] * np.cos(2 * np.pi * f_n * t + theta[n - 1])
        else:
            g += C[n - 1] * np.cos(2 * np.pi * f_n * t + theta[n - 1])

    if show_even and show_odd:
        f = np.arange(K + 1) * f0
    elif show_even:
        f = np.arange(0, 2*K+2, 2) * f0
    elif show_odd:
        f = np.arange(1, 2*K+1, 2) * f0
        f = np.concatenate(([0], f))
    else:
        f = np.array([0])

    C = np.concatenate(([C0], C))
    theta = np.concatenate(([0], theta))
    theta = np.degrees(theta)
    theta = [(i if i != -180 else 0) for i in theta]

    return g, f, C, theta


def update(event):
    N = int(slider_N.val)
    f0 = slider_f0.val
    A = slider_A.val
    g, f, C, theta = calculate_signal(N, f0, A)
    if use_components:
        for i, comp in enumerate(g):
            if i < len(component_lines):
                component_lines[i].set_ydata(comp)
            else:
                component_lines.append(ax[0].plot(t, comp, lw=1)[0])
        while len(component_lines) > len(g):
            component_lines.pop().remove()
        line_g.set_visible(False)
    else:
        line_g.set_ydata(g)
        line_g.set_visible(True)
        for line in component_lines:
            line.remove()
        component_lines.clear()

    scatter_amp.set_offsets(np.column_stack((f, C)))
    scatter_phase.set_offsets(np.column_stack((f, theta)))

    for i, (x, y) in enumerate(zip(f, C)):
        if i < len(lines_amp):
            lines_amp[i].set_data([x, x], [0, y])
        else:
            lines_amp.append(ax[1].plot([x, x], [0, y], color='b', lw=1)[0])

    for i, (x, y) in enumerate(zip(f, theta)):
        if i < len(lines_phase):
            lines_phase[i].set_data([x, x], [0, y])
        else:
            lines_phase.append(ax[2].plot([x, x], [0, y], color='b', lw=1)[0])

    while len(lines_amp) > len(f):
        lines_amp.pop().remove()

    while len(lines_phase) > len(f):
        lines_phase.pop().remove()
        
    fig.canvas.draw_idle()

def reset(event):
    slider_N.reset()
    slider_f0.reset()
    slider_A.reset()
    slider_ylim.reset()
    slider_xlim.reset()
    
    g, f, C, theta = calculate_signal(INIT_N, INIT_F0, INIT_A)

    line_g.set_ydata(g)
    ax[0].set_ylim([-INIT_YLIM, INIT_YLIM])
    ax[1].set_ylim([0, INIT_YLIM])
    ax[2].set_ylim([-90, 0])
    
    scatter_amp.set_offsets(np.column_stack((f, C)))
    for i, (x, y) in enumerate(zip(f, C)):
        lines_amp[i].set_data([x, x], [0, y])

    scatter_phase.set_offsets(np.column_stack((f, theta)))
    for i, (x, y) in enumerate(zip(f, theta)):
        lines_phase[i].set_data([x, x], [0, y])

    ax[0].set_xlim([0, 50 / INIT_XLIM])
    ax[1].set_xlim([0, INIT_XLIM])
    ax[2].set_xlim([0, INIT_XLIM])
    
    fig.canvas.draw_idle()

def update_ylimit(val):
    ylim = slider_ylim.val
    if not lockG:
        ax[0].set_ylim([-ylim, ylim])
    ax[1].set_ylim([0, ylim])
    fig.canvas.draw_idle()

def update_xlimit(val):
    xlim = slider_xlim.val
    if not lockG:
        ax[0].set_xlim([0,50/(xlim)])
    ax[1].set_xlim([0, xlim])
    ax[2].set_xlim([0, xlim])
    fig.canvas.draw_idle()

def toggle_even(event):
    global show_even, show_odd
    show_even = not show_even
    button_toggle_even.color = 'green' if show_even else 'red'  
    update_plot()

def toggle_odd(event):
    global show_odd, show_even
    show_odd = not show_odd
    button_toggle_odd.color = 'green' if show_odd else 'red'  
    update_plot()

def toogle_lockG(event):
    global lockG
    lockG = not lockG
    button_toggle_lock.color = 'green' if lockG else 'red'

def toogle_components(event):
    global use_components
    use_components = not use_components
    button_toggle_componentes.color = 'green' if use_components else 'red'
    update_plot()

def update_plot():
    global use_components
    N = int(slider_N.val)
    f0 = slider_f0.val
    A = slider_A.val
    g, f, C, theta = calculate_signal(N, f0, A)
    if use_components:
        for i, comp in enumerate(g):
            if i < len(component_lines):
                component_lines[i].set_ydata(comp)
            else:
                component_lines.append(ax[0].plot(t, comp, lw=1)[0])
        while len(component_lines) > len(g):
            component_lines.pop().remove()
        line_g.set_visible(False)
    else:
        line_g.set_ydata(g)
        line_g.set_visible(True)
        for line in component_lines:
            line.remove()
        component_lines.clear()

    scatter_amp.set_offsets(np.column_stack((f, C)))
    scatter_phase.set_offsets(np.column_stack((f, theta)))

    for i, (x, y) in enumerate(zip(f, C)):
        if i < len(lines_amp):
            lines_amp[i].set_data([x, x], [0, y])
        else:
            lines_amp.append(ax[1].plot([x, x], [0, y], color='b', lw=1)[0])

    for i, (x, y) in enumerate(zip(f, theta)):
        if i < len(lines_phase):
            lines_phase[i].set_data([x, x], [0, y])
        else:
            lines_phase.append(ax[2].plot([x, x], [0, y], color='b', lw=1)[0])

    while len(lines_amp) > len(f):
        lines_amp.pop().remove()

    while len(lines_phase) > len(f):
        lines_phase.pop().remove()

    fig.canvas.draw_idle()

if __name__ == "__main__":

    Fs = 1000
    Ts = 1 / Fs
    L = 10
    t = np.arange(0, L + Ts, Ts)

    coef_a = lambda a,n: (a / (n * np.pi)) * (np.sin(n * np.pi))
    coef_b = lambda a,n: (a / (n * np.pi)) * (1 - np.cos(n * np.pi))
    coef_C_0 = lambda a: a/2
    
    INIT_N = 10
    INIT_F0 = 5
    INIT_A = 1
    INIT_YLIM = 1.2
    INIT_XLIM = INIT_F0 * INIT_N
    
    g, f, C, theta = calculate_signal(INIT_N, INIT_F0, INIT_A)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.22, hspace=0.3)

    line_g, = ax[0].plot(t, g, lw=2)
    ax[0].set_xlabel('Tempo (s)')
    ax[0].set_ylabel('Amplitude v(t)')

    scatter_amp = ax[1].scatter(f, C, color='b')
    ax[1].set_xlabel('Frequência (Hz)')
    ax[1].set_ylabel('Amplitude V(f)')
    ax[1].set_ylim([0, 0.8])
    lines_amp = [ax[1].plot([x, x], [0, y], color='b', lw=1)[0] for x, y in zip(f, C)]

    scatter_phase = ax[2].scatter(f, theta, color='b')
    ax[2].set_xlabel('Frequência (Hz)')
    ax[2].set_ylabel('Fase θ(f) [graus]')
    ax[2].set_ylim([-90, 0])
    lines_phase = [ax[2].plot([x, x], [0, y], color='b', lw=1)[0] for x, y in zip(f, theta)]

    ax_slider_N = plt.axes([0.25, 0.07, 0.65, 0.03])
    ax_slider_f0 = plt.axes([0.25, 0.02, 0.65, 0.03])
    ax_slider_A = plt.axes([0.25, 0.12, 0.65, 0.03])

    slider_N = Slider(ax_slider_N, 'N', valmin=1, valmax=250, valinit=INIT_N, valstep=1)
    slider_f0 = Slider(ax_slider_f0, 'Fundamental [Hz]', valmin=0.01, valmax=100, valinit=INIT_F0)
    slider_A = Slider(ax_slider_A, 'Amplitude A', valmin=1, valmax=10, valinit=INIT_A)
    ax_slider_ylim = plt.axes([0.01, 0.15, 0.0225, 0.63])
    slider_ylim = Slider(ax_slider_ylim, 'Y', valmin=0.001, valmax=4, valinit=INIT_YLIM, orientation='vertical')
    ax_slider_xlim = plt.axes([0.06, 0.15, 0.0225, 0.63])
    slider_xlim = Slider(ax_slider_xlim, 'X', valmin=0.1, valmax=2500, valinit=INIT_YLIM, orientation='vertical')

    slider_N.on_changed(update)
    slider_f0.on_changed(update)
    slider_A.on_changed(update)
    slider_ylim.on_changed(update_ylimit)
    slider_xlim.on_changed(update_xlimit)

    ax_reset = plt.axes([0.05, 0.93, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset', hovercolor='0.975')
    button_reset.on_clicked(reset)

    ax_toggle_even = plt.axes([0.20, 0.93, 0.1, 0.04])
    button_toggle_even = Button(ax_toggle_even, 'Par', color="green", hovercolor='0.975')
    button_toggle_even.on_clicked(toggle_even)

    ax_toggle_odd = plt.axes([0.35, 0.93, 0.1, 0.04])
    button_toggle_odd = Button(ax_toggle_odd, 'Impar', color="green", hovercolor='0.975')
    button_toggle_odd.on_clicked(toggle_odd)

    ax_toggle_lock = plt.axes([0.50, 0.93, 0.1, 0.04])
    button_toggle_lock = Button(ax_toggle_lock, 'Travar', color="red", hovercolor='0.975')
    button_toggle_lock.on_clicked(toogle_lockG)

    ax_toggle_componentes = plt.axes([0.65, 0.93, 0.15, 0.04])
    button_toggle_componentes = Button(ax_toggle_componentes, 'Componentes', color="red", hovercolor='0.975')
    button_toggle_componentes.on_clicked(toogle_components)

    plt.show()
