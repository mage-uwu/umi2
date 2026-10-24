"""merge_rooms.py: room banks into one (at most 32 rooms, the physics' ROOM_BANK): out.json a.json:K b.json:K ...
takes the first K rooms of each bank (or a single room file as one room)."""
import json, sys
out, rooms = sys.argv[1], []
for spec in sys.argv[2:]:
    f, k = (spec.rsplit(":", 1) + ["999"])[:2] if ":" in spec else (spec, "999")
    d = json.load(open(f)); rs = d["rooms"] if "rooms" in d else [{"name": d.get("name", f), "bounds": d.get("bounds"), "boxes": d["boxes"]}]
    rooms += rs[:int(k)]
assert len(rooms) <= 32, "more than ROOM_BANK rooms"
with open(out, "w") as fo:
    fo.write('{\n  "version": 2,\n  "rooms": [\n' + ",\n".join('    {"name": ' + json.dumps(r["name"]) + ', "bounds": ' + json.dumps(r.get("bounds")) + ', "boxes": [\n'
             + ",\n".join("      " + json.dumps(x) for x in r["boxes"]) + "\n    ]}" for r in rooms) + "\n  ]\n}\n")
print(f"{out}: {len(rooms)} rooms")
