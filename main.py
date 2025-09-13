import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return x + 1

def calculate_method_rectangles(left_bord, right_bord, step, func):
    a = left_bord
    b = left_bord + step
    result = 0.0

    while a < right_bord:
        if b > right_bord:
            b = right_bord

        middle = (a + b) / 2
        h_step = (b - a)

        result += func(middle) * h_step

        a = b
        b += step

    return result

def calculate_a_0(res_integrate):
    a_0 = (1 / np.pi) * res_integrate
    return a_0

def calculate_a_k(k, left_border, right_border, step, func):
    def integrand_func(x):
        return func(x) * np.cos(k * x)

    integral = calculate_method_rectangles(left_border, right_border, step, integrand_func)
    a_k = (1 / np.pi) * integral
    return a_k

def calculate_b_k(k, left_border, right_border, step, func):
    def integrand_func(x):
        return func(x) * np.sin(k * x)

    integral = calculate_method_rectangles(left_border, right_border, step, integrand_func)
    b_k = (1 / np.pi) * integral
    return b_k

def calculate_fourier_coefficients(n_terms, left_border, right_border, step, func):
    a_coeffs = []
    b_coeffs = []

    integral_0 = calculate_method_rectangles(left_border, right_border, step, func)
    a_0 = calculate_a_0(integral_0)

    for k in range(1, n_terms + 1):
        a_k = calculate_a_k(k, left_border, right_border, step, func)
        b_k = calculate_b_k(k, left_border, right_border, step, func)

        a_coeffs.append(a_k)
        b_coeffs.append(b_k)

    return a_0, a_coeffs, b_coeffs

def reconstruct_signal(t, a_0, a_coeffs, b_coeffs):
    result = a_0 / 2 * np.ones_like(t)

    for k in range(1, len(a_coeffs) + 1):
        result += a_coeffs[k - 1] * np.cos(k * t) + b_coeffs[k - 1] * np.sin(k * t)

    return result

def calculate_required_harmonics(left_border, right_border, step_motion, orig_signal, target_error, func=None):
    if func is None:
        func = function

    current_error = 1
    n_terms = 0
    t = np.linspace(left_border, right_border, len(orig_signal))

    while current_error > target_error and n_terms < 100:
        n_terms += 1

        a_0, a_coeffs, b_coeffs = calculate_fourier_coefficients(n_terms, left_border, right_border, step_motion, func)

        reconstructed_signal = reconstruct_signal(t, a_0, a_coeffs, b_coeffs)

        avg_approx_error = np.mean((orig_signal - reconstructed_signal) ** 2)
        relative_error = avg_approx_error / np.mean(orig_signal ** 2)
        current_error = relative_error

    return n_terms, a_0, a_coeffs, b_coeffs

def calculate_trend(left_border, right_border, step, func):
    integral_0 = calculate_method_rectangles(left_border, right_border, step, func)
    a_0 = calculate_a_0(integral_0)
    trend = a_0 / 2
    return trend


if __name__ == "__main__":

    left_border = -np.pi
    right_border = np.pi
    step_motion = 1 * 10 ** -4
    t = np.linspace(left_border, right_border, 1000)

    target_error = 0.1

    original_signal = function(t)

    # print("Исходная функция")
    N_optimal, a_0_f, a_coeffs_f, b_coeffs_f = calculate_required_harmonics(left_border, right_border, step_motion, original_signal, target_error, function)
    print(f"Для исходной функции потребовалось N = {N_optimal} гармоник\n")

    reconstructed_f = reconstruct_signal(t, a_0_f, a_coeffs_f, b_coeffs_f)


    # print("Функция без тренда")
    trend = calculate_trend(left_border, right_border, step_motion, function)
    print(f"Тренд: {trend:.6f}")

    def g_function(x):
        return function(x) - trend

    g_signal = g_function(t)

    M_optimal, a_0_g, a_coeffs_g, b_coeffs_g = calculate_required_harmonics(left_border, right_border, step_motion, g_signal, target_error, g_function)
    print(f"Для функции без тренда потребовалось M = {M_optimal} гармоник")

    reconstructed_g = reconstruct_signal(t, a_0_g, a_coeffs_g, b_coeffs_g)
    reconstructed_f_from_g = trend + reconstructed_g


    plt.figure(figsize=(10, 7))

    plt.subplot(2, 1, 1)
    plt.plot(t, original_signal, 'b-', linewidth=2, label=f'Исходная функция f(t)')
    plt.plot(t, reconstructed_f, 'r--', linewidth=1.5, label=f'Аппроксимация (N={N_optimal})')
    plt.title(f'Точность: {target_error * 100}%\n\nАппроксимация исходной функции', fontsize=14)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, original_signal, 'b-', linewidth=2, label='Исходная функция f(t)')
    plt.plot(t, reconstructed_f_from_g, 'g--', linewidth=1.5, label=f'Аппроксимация (M={M_optimal})')
    plt.axhline(y=trend, color='m', linestyle=':', linewidth=2, label='Тренд')
    plt.title('Аппроксимация через функцию без тренда', fontsize=14)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()