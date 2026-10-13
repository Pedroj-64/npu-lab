"""Offline test of the lines kept word for word through a compaction. usage: python tests/test_pins.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pins

old = [{"role": "user", "content": "My project is called Nebula, it listens on port 8431 and my cat is called Zeta. Reply only with OK."},
       {"role": "assistant", "content": "OK, port 8432"},
       {"role": "user", "content": "Name three uses of the fish shell."},
       {"role": "user", "content": "Edit ~/code/app/server.py so it reads config.toml\nthanks"}]
p = pins.pin(old)
assert p[0].startswith("My project is called Nebula") and "8431" in p[0], p                    # the user's line, not the assistant's 8432
assert any("server.py" in l for l in p) and not any("fish shell" in l for l in p), p
assert not any("8432" in l for l in p)
# a second compaction receives the first block inside the notes message: facts carry over once, not twice
again = pins.pin([{"role": "user", "content": "[Notes of the earlier conversation]\n" + pins.block(p) + "* Project: Nebula, port 8431 (the model's summary bullet)"}] + old)
assert again == p, again
many = pins.pin([{"role": "user", "content": f"port {i} is open"} for i in range(40)])
assert len(many) == pins.MAX_LINES and many[0] == "port 0 is open" and many[-1] == "port 39 is open", many      # oldest and newest are kept
assert pins.pin([{"role": "user", "content": "hello there, how are you today"}]) == [] and pins.block([]) == ""
print("pins ok")
