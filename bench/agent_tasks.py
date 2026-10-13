"""Task bank for the agent evaluation and for the labelled "is it finished?" data of the decision layer (src/decide.py).
Two fixtures: "base" (the small shop project of edit_eval.py + an unused legacy.py) and "shop" (a second project WITH its own unittest suite, so the
agent can use RUN). Each task: instruction in English and Spanish, a check (code run inside the project folder; exit 0 = solved) and a reference solution
(file -> new content, None = delete). `python bench/agent_tasks.py` verifies every task: the starting project FAILS its check, the reference PASSES it.
Tasks 1-5 are the original ones of agent_eval.py; their reference is obtained by running that file's scripted perfect agent."""
import os, subprocess, sys, tempfile, shutil, itertools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import edit_eval as E

LEGACY = '"""Old helpers nobody uses."""\n\n\ndef old():\n    return 0\n'
def _base():                               # exactly the project agent_eval.py builds: the five shop files + 8 unrelated filler modules + legacy.py
    d = tempfile.mkdtemp()
    try:
        files = dict(E.project(d, 8))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return dict(files, **{"legacy.py": LEGACY})


BASE = _base()
SHOP = {
    "inventory.py": 'class Inventory:\n    """Units in stock per sku."""\n\n    def __init__(self):\n        self.stock = {}\n\n    def add(self, sku, n):\n        """Put n more units of sku in stock."""\n        self.stock[sku] = self.stock.get(sku, 0) + n\n\n'
                    '    def remove(self, sku, n):\n        """Take n units of sku out of stock; ValueError when there are not enough."""\n        if self.stock.get(sku, 0) <= n:\n            raise ValueError("not enough stock")\n        self.stock[sku] -= n\n\n'
                    '    def count(self, sku):\n        """Units of sku in stock."""\n        return self.stock.get(sku, 0)\n',
    "orders.py": 'class Order:\n    """Lines of (sku, qty, unit price)."""\n\n    def __init__(self):\n        self.lines = []\n\n    def add_line(self, sku, qty, price):\n        """Add a line to the order."""\n        self.lines.append((sku, qty, price))\n\n'
                 '    def total(self):\n        """Sum of the line totals."""\n        return round(sum(price for _, qty, price in self.lines), 2)\n',
    "tests/__init__.py": "",
    "tests/test_inventory.py": 'import unittest\nfrom inventory import Inventory\n\n\nclass T(unittest.TestCase):\n    def test_add_count(self):\n        i = Inventory(); i.add("a", 3); i.add("a", 2)\n        self.assertEqual(i.count("a"), 5)\n\n'
                               '    def test_remove(self):\n        i = Inventory(); i.add("a", 5); i.remove("a", 2)\n        self.assertEqual(i.count("a"), 3)\n\n    def test_remove_too_many(self):\n        i = Inventory(); i.add("a", 1)\n        with self.assertRaises(ValueError):\n            i.remove("a", 2)\n',
    "tests/test_orders.py": 'import unittest\nfrom orders import Order\n\n\nclass T(unittest.TestCase):\n    def test_total_single_units(self):\n        o = Order(); o.add_line("a", 1, 2.5); o.add_line("b", 1, 1.0)\n        self.assertEqual(o.total(), 3.5)\n',
}
TEST_COMMAND = ["python", "-m", "unittest", "discover", "-t", ".", "-s", "tests"]      # the command a user would configure for RUN on the "shop" fixture
FIXTURES = {"base": BASE, "shop": SHOP}


def sub(text, a, b, n=1):
    assert text.count(a) == n, (a, text.count(a))
    return text.replace(a, b)


def head(src, a, b):                      # one replacement on a copy of a fixture file
    return sub(src, a, b)


B = BASE
TASKS = {}                                # id -> dict(fixture, en, es, check, ref)


def task(tid, fixture, en, es, check, ref):
    TASKS[tid] = dict(fixture=fixture, en=en, es=es, check=check, ref=ref)


# --- 1-5: the original agent_eval tasks (reference filled in by build_original_refs below) ---
import agent_eval as L  # noqa: E402  (its top level only defines things; it imports this file lazily, inside functions)
ES = {1: "En util.py haz que clamp lance ValueError cuando lo sea mayor que hi.",
      2: "Crea un archivo nuevo stats.py con una función average(nums) que devuelva la media de una lista de números, y 0.0 si la lista está vacía.",
      3: "Añade un método total_value() a Store (suma de qty * price de todos los artículos) y haz que main en cli.py imprima una línea final 'TOTAL: ' seguida de ese valor con dos decimales.",
      4: "Borra el archivo legacy.py: nada lo usa.",
      5: "Renombra la función format_line a format_item en todos los sitios donde aparece (report.py y cli.py)."}
for k, (txt, chk) in L.TASKS.items():
    task(k, "base", txt, ES[k], chk, None)

task(6, "base", "In util.py add a function truncate(text, n) that returns text unchanged when it has at most n characters, otherwise the first n characters followed by '...'.",
     "En util.py añade una función truncate(text, n) que devuelva el texto igual si tiene n caracteres o menos, y si no, los primeros n caracteres seguidos de '...'.",
     "from util import truncate\nassert truncate('hello', 5) == 'hello' and truncate('hello world', 5) == 'hello...' and truncate('', 3) == ''",
     {"util.py": B["util.py"] + '\n\ndef truncate(text, n):\n    """Text cut to n characters, with ... when something was cut."""\n    return text if len(text) <= n else text[:n] + "..."\n'})
task(7, "base", "Add a method remove(name) to Store that deletes the item called name and returns True, or returns False when there is no such item.",
     "Añade a Store un método remove(name) que borre el artículo llamado name y devuelva True, o devuelva False si no existe.",
     "from store import Store\ns = Store(); s.add('a', 1, 1.0)\nassert s.remove('a') is True and s.get('a') is None and s.remove('a') is False",
     {"store.py": B["store.py"] + '\n    def remove(self, name):\n        """Delete the item called name; True if it existed."""\n        return self._items.pop(name, None) is not None\n'})
task(8, "base", "Add a method count() to Store that returns the total number of units, which is the sum of qty over all items.",
     "Añade a Store un método count() que devuelva el total de unidades, es decir, la suma de qty de todos los artículos.",
     "from store import Store\ns = Store(); assert s.count() == 0\ns.add('a', 2, 1.0); s.add('b', 3, 1.0)\nassert s.count() == 5",
     {"store.py": B["store.py"] + '\n    def count(self):\n        """Total units over all items."""\n        return sum(i.qty for i in self._items.values())\n'})
task(9, "base", "Add a property total to the Item dataclass in models.py that returns qty * price.",
     "Añade a la dataclass Item de models.py una propiedad total que devuelva qty * price.",
     "from models import Item\nassert Item('a', 3, 1.5).total == 4.5",
     {"models.py": B["models.py"] + "\n    @property\n    def total(self):\n        return self.qty * self.price\n"})
task(10, "base", "In pricing.py add a function bulk_price(price, qty) that returns price * qty, with a 10% discount on the whole amount when qty is 10 or more, rounded to 2 decimals.",
     "En pricing.py añade una función bulk_price(price, qty) que devuelva price * qty, con un 10% de descuento sobre el total si qty es 10 o más, redondeado a 2 decimales.",
     "from pricing import bulk_price\nassert bulk_price(2.0, 5) == 10.0 and bulk_price(2.0, 10) == 18.0",
     {"pricing.py": B["pricing.py"] + '\n\ndef bulk_price(price, qty):\n    """Total for qty units, 10% off from 10 units."""\n    total = price * qty\n    return round(total * 0.9 if qty >= 10 else total, 2)\n'})
task(11, "base", "In report.py add a function table(items) that takes a list of Item objects and returns their format_line texts joined with newlines.",
     "En report.py añade una función table(items) que reciba una lista de objetos Item y devuelva sus textos de format_line unidos con saltos de línea.",
     "from models import Item\nfrom report import table\nassert table([Item('a', 1, 2.0), Item('b', 2, 1.0)]) == 'a: 1 x 2.00\\nb: 2 x 1.00'",
     {"report.py": B["report.py"] + '\n\ndef table(items):\n    """All items, one line each."""\n    return "\\n".join(format_line(i.name, i.qty, i.price) for i in items)\n'})
task(12, "base", "Make main in cli.py accept an optional '--sort' flag; when present print the items sorted by name ignoring case (Ink, Pen... order: book, Ink, Pen).",
     "Haz que main en cli.py acepte una bandera opcional '--sort'; si está, imprime los artículos ordenados por nombre sin distinguir mayúsculas (orden: book, Ink, Pen).",
     "import io, contextlib, cli\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    cli.main(['--sort'])\nnames = [l.split(':')[0] for l in b.getvalue().splitlines()]\nassert names == ['book', 'Ink', 'Pen'], names",
     {"cli.py": sub(B["cli.py"], "    for it in store.items():\n", '    items = store.items()\n    if "--sort" in argv:\n        items = sorted(items, key=lambda i: i.name.lower())\n    for it in items:\n')})
task(13, "base", "Make slug in util.py also strip spaces at both ends and collapse runs of spaces into a single dash.",
     "Haz que slug en util.py también quite los espacios de los extremos y reduzca varios espacios seguidos a un solo guion.",
     "from util import slug\nassert slug('  Hello   World ') == 'hello-world' and slug('a b') == 'a-b'",
     {"util.py": sub(B["util.py"], '    return text.lower().replace(" ", "-")', '    return "-".join(text.lower().split())')})
task(14, "base", "Create config.py with TAX_RATE = 0.21 and make price_with_tax in pricing.py use TAX_RATE as the default value of rate.",
     "Crea config.py con TAX_RATE = 0.21 y haz que price_with_tax de pricing.py use TAX_RATE como valor por defecto de rate.",
     "import config, pricing\nassert config.TAX_RATE == 0.21\nconfig_default = pricing.price_with_tax.__defaults__[0]\nassert config_default == config.TAX_RATE\nsrc = open('pricing.py').read()\nassert 'TAX_RATE' in src and 'rate=0.21' not in src",
     {"config.py": "TAX_RATE = 0.21\n", "pricing.py": sub("from config import TAX_RATE\n\n\n" + B["pricing.py"], "rate=0.21", "rate=TAX_RATE")})
task(15, "base", "Make Store.add raise ValueError when qty is negative.",
     "Haz que Store.add lance ValueError cuando qty sea negativa.",
     "from store import Store\ns = Store(); s.add('a', 0, 1.0)\ntry:\n    s.add('b', -1, 1.0)\nexcept ValueError:\n    assert s.get('b') is None\nelse:\n    raise SystemExit('no ValueError')",
     {"store.py": sub(B["store.py"], '        self._items[name] = Item(name, qty, price)', '        if qty < 0:\n            raise ValueError("qty must not be negative")\n        self._items[name] = Item(name, qty, price)')})
task(16, "base", "Move clamp out of util.py into a new file mathx.py, and keep 'from util import clamp' working by importing it in util.py.",
     "Mueve clamp de util.py a un archivo nuevo mathx.py, y haz que 'from util import clamp' siga funcionando importándola en util.py.",
     "import mathx, util\nassert mathx.clamp(5, 0, 3) == 3 and util.clamp(-1, 0, 3) == 0\nassert 'def clamp' not in open('util.py').read() and 'def clamp' in open('mathx.py').read()",
     {"mathx.py": 'def clamp(x, lo, hi):\n    """Limit x to the range lo..hi."""\n    return max(lo, min(hi, x))\n',
      "util.py": 'from mathx import clamp  # noqa: F401\n\n\ndef slug(text):\n    """Lowercase text with spaces turned into dashes."""\n    return text.lower().replace(" ", "-")\n'})
task(17, "base", "Rename the method Store.items to all_items everywhere it is defined or used (store.py and cli.py).",
     "Renombra el método Store.items a all_items en todos los sitios donde se define o se usa (store.py y cli.py).",
     "import io, contextlib, cli, store\nassert hasattr(store.Store, 'all_items') and not hasattr(store.Store, 'items')\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    cli.main([])\nassert 'Pen' in b.getvalue()",
     {"store.py": sub(B["store.py"], "    def items(self):", "    def all_items(self):"), "cli.py": sub(B["cli.py"], "store.items()", "store.all_items()")})
task(18, "base", "Add an optional parameter cur='' to format_line in report.py; when cur is not empty, write it right before the price (format_line('a', 2, 1.5, '$') gives 'a: 2 x $1.50'). Keep the old behaviour when cur is empty.",
     "Añade a format_line de report.py un parámetro opcional cur='' ; si no está vacío, escríbelo justo antes del precio (format_line('a', 2, 1.5, '$') da 'a: 2 x $1.50'). Con cur vacío el resultado no cambia.",
     "from report import format_line\nassert format_line('a', 2, 1.5, '$') == 'a: 2 x $1.50' and format_line('a', 2, 1.5) == 'a: 2 x 1.50'",
     {"report.py": 'def format_line(name, qty, price, cur=""):\n    """One text line for an item."""\n    return f"{name}: {qty} x {cur}{price:.2f}"\n'})
task(19, "base", "Make price_with_tax raise ValueError when price is negative.",
     "Haz que price_with_tax lance ValueError cuando price sea negativo.",
     "from pricing import price_with_tax\nassert price_with_tax(100) == 121.0\ntry:\n    price_with_tax(-1)\nexcept ValueError:\n    pass\nelse:\n    raise SystemExit('no ValueError')",
     {"pricing.py": sub(B["pricing.py"], '    """Price including tax."""\n', '    """Price including tax."""\n    if price < 0:\n        raise ValueError("price must not be negative")\n')})
task(20, "base", "In cli.py add a function count_items(store) that returns how many different items the store has, and make main print a first line 'COUNT: n' with it.",
     "En cli.py añade una función count_items(store) que devuelva cuántos artículos distintos tiene la tienda, y haz que main imprima como primera línea 'COUNT: n' con ese valor.",
     "import io, contextlib, cli, store\ns = store.Store(); cli.load(s)\nassert cli.count_items(s) == 3\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    cli.main([])\nassert b.getvalue().splitlines()[0] == 'COUNT: 3', b.getvalue()",
     {"cli.py": sub(sub(B["cli.py"], "def main(argv):", 'def count_items(store):\n    """How many different items there are."""\n    return len(store.items())\n\n\ndef main(argv):'),
                    "    for it in store.items():\n", '    print(f"COUNT: {count_items(store)}")\n    for it in store.items():\n')})
# --- shop fixture (own tests; the visible tests do NOT cover what the checks ask) ---
task(21, "shop", "There is a bug: removing exactly all the units of a sku from Inventory raises ValueError, but it must work (only removing MORE than the stock is an error). Fix it.",
     "Hay un error: quitar de Inventory exactamente todas las unidades de un sku lanza ValueError, pero debe funcionar (solo es error quitar MÁS de lo que hay). Corrígelo.",
     "from inventory import Inventory\ni = Inventory(); i.add('a', 2); i.remove('a', 2)\nassert i.count('a') == 0\ntry:\n    i.remove('a', 1)\nexcept ValueError:\n    pass\nelse:\n    raise SystemExit('no ValueError')",
     {"inventory.py": sub(SHOP["inventory.py"], "<= n:", "< n:")})
task(22, "shop", "Add a method low_stock(threshold) to Inventory that returns the sorted list of skus whose count is below threshold.",
     "Añade a Inventory un método low_stock(threshold) que devuelva la lista ordenada de skus cuyo count esté por debajo de threshold.",
     "from inventory import Inventory\ni = Inventory(); i.add('b', 1); i.add('a', 2); i.add('c', 9)\nassert i.low_stock(3) == ['a', 'b'] and i.low_stock(1) == []",
     {"inventory.py": SHOP["inventory.py"] + '\n    def low_stock(self, threshold):\n        """Sorted skus with fewer than threshold units."""\n        return sorted(s for s, n in self.stock.items() if n < threshold)\n'})
task(23, "shop", "Order.total is wrong when a line has quantity greater than 1: it must add qty * price for every line. Fix it.",
     "Order.total da un resultado incorrecto cuando una línea tiene cantidad mayor que 1: debe sumar qty * price de cada línea. Corrígelo.",
     "from orders import Order\no = Order(); o.add_line('a', 3, 2.0); o.add_line('b', 1, 0.5)\nassert o.total() == 6.5",
     {"orders.py": sub(SHOP["orders.py"], "sum(price for _, qty, price in self.lines)", "sum(qty * price for _, qty, price in self.lines)")})
task(24, "shop", "Make Order.add_line raise ValueError when qty is not positive, and add a test for it in tests/test_orders.py.",
     "Haz que Order.add_line lance ValueError cuando qty no sea positiva, y añade una prueba para ello en tests/test_orders.py.",
     "from orders import Order\no = Order()\nfor bad in (0, -2):\n    try:\n        o.add_line('a', bad, 1.0)\n    except ValueError:\n        pass\n    else:\n        raise SystemExit('no ValueError')\nassert o.total() == 0\nsrc = open('tests/test_orders.py').read()\nassert src.count('def test_') >= 2 and 'ValueError' in src",
     {"orders.py": sub(SHOP["orders.py"], '        self.lines.append((sku, qty, price))', '        if qty <= 0:\n            raise ValueError("qty must be positive")\n        self.lines.append((sku, qty, price))'),
      "tests/test_orders.py": SHOP["tests/test_orders.py"] + '\n    def test_bad_qty(self):\n        with self.assertRaises(ValueError):\n            Order().add_line("a", 0, 1.0)\n'})
task(25, "shop", "Rename Inventory.count to quantity everywhere (inventory.py and the tests) and make sure the test suite still passes.",
     "Renombra Inventory.count a quantity en todos los sitios (inventory.py y las pruebas) y comprueba que la batería de pruebas sigue pasando.",
     "import inventory, subprocess, sys\nassert hasattr(inventory.Inventory, 'quantity') and not hasattr(inventory.Inventory, 'count')\ni = inventory.Inventory(); i.add('a', 2)\nassert i.quantity('a') == 2\nr = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-t', '.', '-s', 'tests'], capture_output=True, text=True)\nassert r.returncode == 0, r.stderr[-300:]",
     {"inventory.py": sub(SHOP["inventory.py"], "def count(self, sku)", "def quantity(self, sku)"),
      "tests/test_inventory.py": SHOP["tests/test_inventory.py"].replace("i.count(", "i.quantity(")})


def write_project(folder, fixture):
    for n, t in FIXTURES[fixture].items():
        p = os.path.join(folder, n); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(t)


def snapshot(folder):
    out = {}
    for d, subs, fs in os.walk(folder):
        subs[:] = [s for s in subs if s != "__pycache__"]
        for f in fs:
            if not f.endswith(".pyc"):
                out[os.path.relpath(os.path.join(d, f), folder)] = open(os.path.join(d, f)).read()
    return out


def apply_files(folder, files):
    for n, t in files.items():
        p = os.path.join(folder, n)
        if t is None:
            os.path.exists(p) and os.remove(p)
        else:
            os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(t)


def run_check(folder, tid):
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.')\n" + TASKS[tid]["check"]], cwd=folder, capture_output=True, text=True, timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1][:160]


def original_refs():
    """Reference of tasks 1-5 = the project left by the scripted perfect agent of agent_eval.py."""
    import agent_eval as A
    refs = {}
    for tid in range(1, 6):
        root = tempfile.mkdtemp(dir=os.path.expanduser("~"))
        try:
            A.make_project(root)
            script = iter(A.SCRIPT[tid])
            A.run_task  # noqa: B018 (harness is exercised by agent_eval --check; here we only need the final files)
            import agent
            last = True
            for reply in script:
                r = agent.run(root, reply, confirmed=True, last_ok=last); last = r["ok"]
            snap = snapshot(root)
            base = FIXTURES["base"]
            refs[tid] = {n: snap.get(n) for n in set(snap) | set(base) if snap.get(n) != base.get(n)}
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return refs


def fill_original_refs():
    for tid, ref in original_refs().items():
        TASKS[tid]["ref"] = ref


def verify():
    fill_original_refs()
    for tid, t in sorted(TASKS.items()):
        root = tempfile.mkdtemp(dir=os.path.expanduser("~"))
        try:
            write_project(root, t["fixture"])
            start, why = run_check(root, tid)
            assert not start, f"task {tid}: the starting project already passes its check"
            apply_files(root, t["ref"])
            ok, why = run_check(root, tid)
            assert ok, f"task {tid}: reference fails its check: {why}"
            if t["fixture"] == "shop":
                r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-t", ".", "-s", "tests"], cwd=root, capture_output=True, text=True)
                assert r.returncode == 0, f"task {tid}: the project's own tests fail after the reference: {r.stderr[-200:]}"
        finally:
            shutil.rmtree(root, ignore_errors=True)
    # the visible suite of the shop fixture passes at the start (it does not see the bugs of tasks 21 and 23)
    root = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    try:
        write_project(root, "shop")
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-t", ".", "-s", "tests"], cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-300:]
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"agent tasks ok: {len(TASKS)} tasks, each fails at the start and passes with its reference")


if __name__ == "__main__":
    verify()
