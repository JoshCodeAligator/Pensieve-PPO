import json

class LayerSizeDB:
    def __init__(self, path):
        with open(path, "r") as f:
            self.db = json.load(f)

    def get_sizes_bits(self, video_id: str, k: int):
        recs = self.db[video_id]["chunks"]
        n = len(recs)
        if n == 0:
            raise ValueError(f"No chunks defined for {video_id} in sizes.json")
        # wrap the requested index so short test videos don't crash long loops
        kw = int(k % n)
        rec = recs[kw]
        return rec["S_BL"], rec["S_E1"], rec["S_E2"]

    def num_chunks(self, vid: str) -> int:
        return len(self.db[vid]["chunks"])