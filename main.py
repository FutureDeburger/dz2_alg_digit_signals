import numpy as np
# import matplotlib.pyplot as plt

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

def calculate_a_k(k, left_border, right_border, step, func=function):

    def integrand_func(x):
        return func(x) * np.cos(k * x)

    integral = calculate_method_rectangles(left_border, right_border, step, integrand_func)
    a_k = (1 / np.pi) * integral

    return a_k

def calculate_b_k(k, left_border, right_border, step, func=function):

    def integrand_func(x):
        return func(x) * np.sin(k * x)

    integral = calculate_method_rectangles(left_border, right_border, step, integrand_func)
    b_k = (1 / np.pi) * integral
    return b_k

def calculate_fourier_coefficients(n_terms, left_border, right_border, step, func=function):
    a_coeffs = []
    b_coeffs = []

    integral_0 = calculate_method_rectangles(left_border, right_border, step, func)
    a_0 = calculate_a_0(integral_0)

    for k in range(1, n_terms + 1):
        a_k = calculate_a_k(k, left_border, right_border, step)
        b_k = calculate_b_k(k, left_border, right_border, step)

        a_coeffs.append(a_k)
        b_coeffs.append(b_k)

    return a_0, a_coeffs, b_coeffs

def find_a_0(left_border, right_border, step, func=function):
    integral_0 = calculate_method_rectangles(left_border, right_border, step, func)
    a_0 = calculate_a_0(integral_0)
    return a_0

def reconstruct_signal(t, a_0, a_coeffs, b_coeffs):
    result = a_0 / 2 * np.ones_like(t)

    for k in range(1, len(a_coeffs) + 1):
        result += a_coeffs[k - 1] * np.cos(k * t) + b_coeffs[k - 1] * np.sin(k * t)

    return result

def calculate_required_harmonics(left_border, right_border, step_motion, orig_signal, target_error=0.1):

    current_error = 1
    n_terms = 0
    n_values = []
    error_values = []

    while current_error > target_error and n_terms < 100:
        n_terms += 1
        # print(f"Проверяем N = {n_terms}")

        a_0, a_coeffs, b_coeffs = calculate_fourier_coefficients(n_terms, left_border, right_border, step_motion)
        reconstructed_signal = reconstruct_signal(t, a_0, a_coeffs, b_coeffs)

        avg_approx_error = np.mean((orig_signal - reconstructed_signal) ** 2)
        relative_error = avg_approx_error / np.mean(orig_signal ** 2)

        current_error = relative_error
        n_values.append(n_terms)
        error_values.append(current_error)
        # print(f"N = {n_terms}, Ошибка = {current_error:.4f} ({current_error * 100:.2f}%)")
    # print(f"\nРезультат: Для достижения ошибки ≤10% потребовалось N = {n_terms} гармоник")
    return n_terms

def function_without_trend(x, trend):
    return function(x) - trend


if __name__ == "__main__":
    # step_motion = 1 * 10 ** -4
    # n_coeffs = 5
    #
    # a0, an, bn = calculate_fourier_coefficients(n_coeffs, -np.pi, np.pi, step_motion)

    # n_periods = 2



    left_border = -np.pi
    right_border = np.pi
    step_motion = 1 * 10 ** -3
    t = np.linspace(-np.pi, np.pi, 1000)
    original_signal = function(t)

    number_harmonics_for_function = calculate_required_harmonics(left_border, right_border, step_motion, original_signal)
    print(number_harmonics_for_function)


    a_0 = find_a_0(left_border, right_border, step_motion)
    my_trend = a_0 / 2
    print(my_trend)

    # calculate_fourier_coefficients(0, left_border, right_border, step_motion, function_without_trend(my_trend))



    # n_terms = 10
    #
    # a_0, a_coeffs, b_coeffs = calculate_fourier_coefficients(n_terms, left_border, right_border, step_motion)
    #
    #
    #
    #
    #
    # reconstructed_signal = reconstruct_signal(t, a_0, a_coeffs, b_coeffs)
    #
    #
    #
    #
    #
    # trend_function = a_0 / 2







    # print("Коэффициенты Фурье треугольного сигнала:")
    # print(f"a_0 = {a_0:.6f}")
    # print("\nКоэффициенты a_k:")
    # for k, a_k in enumerate(a_coeffs, 1):
    #     print(f"a_{k} = {a_k:.6f}")
    #
    # print("\nКоэффициенты b_k:")
    # for k, b_k in enumerate(b_coeffs, 1):
    #     print(f"b_{k} = {b_k:.6f}")




    # plt.figure(figsize=(7, 7))
    #
    # plt.plot(t, original_signal, 'b-', linewidth=2.5, label='Исходный треугольный сигнал', alpha=0.8)
    # plt.plot(t, reconstructed_signal, 'r--', linewidth=2, label=f'Ряд Фурье ({n_terms} гармоник)')
    #
    # plt.title(f'..., {n_terms} гармоник(и)', fontsize=16)
    # plt.grid(True, alpha=0.3)
    # plt.ylim(-7, 7)
    # plt.xlim(-7, 7)
    #
    # plt.tight_layout()
    # plt.show()
