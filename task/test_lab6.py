import io
import math
from contextlib import redirect_stdout
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import task


def test_full_comparison():
    plots = {}

    def capture_plot(path, **kwargs):
        plots[path.rsplit("/", 1)[-1]] = [
            (ax.get_title(), list(ax.lines)) for ax in plt.gcf().axes
        ]

    output = io.StringIO()
    with patch("utils.plt.savefig", side_effect=capture_plot), redirect_stdout(output):
        task.main()

    names = ["Явная (крест)", "Неявная", "Явная (1-й пор. НУ)", "Неявная (1-й пор. НУ)"]
    text = output.getvalue()
    assert names[-1] in text, "Неявная схема с первым порядком НУ пропущена"
    assert "Погрешности в разные моменты времени" in text, "Нет таблицы ошибок по времени"
    table = text.split("Погрешности в разные моменты времени", 1)[1]
    solvers = [task.solve_explicit, task.solve_implicit,
               task.solve_explicit_order1, task.solve_implicit_order1]
    for name, solver in zip(names, solvers):
        rows = [line[24:].split() for line in table.splitlines() if line[:24].strip() == name]
        assert len(rows) == 4, (name, rows)
        x, times, layers, _ = solver()
        for row, k in zip(rows, [0, 25, 50, 100]):
            t, max_error, rmse = map(float, row)
            assert math.isclose(t, times[k], abs_tol=1e-12)
            errors = [u - math.sin(xx - times[k]) for xx, u in zip(x, layers[k])]
            assert math.isclose(max_error, max(map(abs, errors)), rel_tol=1e-6, abs_tol=1e-15)
            assert math.isclose(rmse, math.sqrt(sum(e * e for e in errors) / len(errors)),
                                rel_tol=1e-6, abs_tol=1e-15)

    for _, lines in plots["plot_compare.png"]:
        assert len(lines) == 5, "Нужны четыре схемы и точное решение"
    profiles = plots["plot_grid_2x2.png"]
    assert [title for title, _ in profiles] == names
    assert all(len(lines) == 8 for _, lines in profiles)
    for _, lines in profiles:
        for numerical, exact in zip(lines[::2], lines[1::2]):
            assert exact.get_linestyle() == "--"
            assert exact.get_color() != numerical.get_color(), "Пунктир сливается с численным решением"
        assert lines[1].get_label() == "Точное решение"
    for _, lines in plots["plot_convergence.png"]:
        assert len(lines) == 4, "Сходимость должна исследоваться для всех четырех схем"


if __name__ == "__main__":
    test_full_comparison()
    print("Проверка lab6 пройдена")
