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

def calculate_required_harmonics(left_border, right_border, step_motion, orig_signal, target_error=0.05, func=None):
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

        # print(f"N = {n_terms}, Ошибка = {current_error:.4f} ({current_error * 100:.2f}%)")
    return n_terms

def calculate_trend(left_border, right_border, step, func):
    integral_0 = calculate_method_rectangles(left_border, right_border, step, func)
    a_0 = calculate_a_0(integral_0)
    trend = a_0 / 2
    return trend

def function_without_trend(x, trend):
    return function(x) - trend


if __name__ == "__main__":

    left_border = -np.pi
    right_border = np.pi
    step_motion = 1 * 10 ** -4
    t = np.linspace(-np.pi, np.pi, 1000)
    original_signal = function(t)

    number_harmonics_for_my_function = calculate_required_harmonics(left_border, right_border, step_motion, original_signal, func=function)
    # print(f"Для исходной функции потребовалось N = {number_harmonics_for_my_function} гармоник")

    my_trend = calculate_trend(left_border, right_border, step_motion, function)
    # print(f"Тренд: {my_trend:.6f}")


    def g_function(x):
        return function_without_trend(x, my_trend)

    # func_without_trend_signal = g_function(t)

    # a_0, a_k, b_k = calculate_fourier_coefficients(3, left_border, right_border, step_motion, func_without_trend_signal)
    #
    # print(a_k)

    # number_harmonics_for_function_without_trend = calculate_required_harmonics(left_border, right_border, step_motion, func_without_trend_signal, func=g_function)
    # print(f"Для функции без тренда потребовалось M = {number_harmonics_for_function_without_trend} гармоник")

    # Проверим первые 5 коэффициентов вручную
    print("\nПроверка коэффициентов для g(t) = t:")
    print("Теоретические значения: a_n = 0, b_n = 2*(-1)^(n+1)/n")

    for n in range(1, 6):
        a_n = calculate_a_k(n, left_border, right_border, step_motion, g_function)
        b_n = calculate_b_k(n, left_border, right_border, step_motion, g_function)
        theoretical_b_n = 2 * (-1) ** (n + 1) / n

        print(f"n={n}: a_{n} = {a_n:.8f}, b_{n} = {b_n:.8f} (теор.: {theoretical_b_n:.8f})")
        print(f"   Ошибка b_{n}: {abs(b_n - theoretical_b_n):.8f}")

    # 3. Проверим a₀ для g(t) - должен быть близок к 0
    integral_g = calculate_method_rectangles(left_border, right_border, step_motion, g_function)
    a_0_g = calculate_a_0(integral_g)
    print(f"\na₀ для g(t): {a_0_g:.10f} (должен быть ~0)")

    # 4. Проверим сходимость для g(t) с выводом ошибки
    print("\nПодбор гармоник для g(t):")
    func_without_trend_signal = g_function(t)

    current_error = 1
    n_terms = 0
    error_history = []

    while current_error > 0.05 and n_terms < 50:
        n_terms += 1

        a_0, a_coeffs, b_coeffs = calculate_fourier_coefficients(
            n_terms, left_border, right_border, step_motion, g_function
        )

        reconstructed_signal = reconstruct_signal(t, a_0, a_coeffs, b_coeffs)

        avg_approx_error = np.mean((func_without_trend_signal - reconstructed_signal) ** 2)
        relative_error = avg_approx_error / np.mean(func_without_trend_signal ** 2)

        current_error = relative_error
        error_history.append((n_terms, current_error))

        if n_terms <= 10 or n_terms % 5 == 0:
            print(f"M = {n_terms:2d}, Ошибка = {current_error:.6f} ({current_error * 100:.2f}%)")

    print(f"\nДля функции без тренда потребовалось M = {n_terms} гармоник")

    # 5. Для сравнения - исходная функция
    print("\nПодбор гармоник для f(t):")
    number_harmonics_for_my_function = calculate_required_harmonics(
        left_border, right_border, step_motion, original_signal, func=function
    )
    print(f"Для исходной функции потребовалось N = {number_harmonics_for_my_function} гармоник")

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
