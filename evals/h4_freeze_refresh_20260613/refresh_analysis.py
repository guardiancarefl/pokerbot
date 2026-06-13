#!/usr/bin/env python3
"""H4 freeze data-refresh: recompute every FIELD_DOSSIER constant behind
T1-T8 on the current n>=500 corpus, reproducing the dossier methodology
(sqlite-only, raw-record sessions, non-hero frame_diff rows).

Methodology mirrors:
  - src/nlhe/opponent_db/db.py::_session_vpip (VPIP convention)
  - FIELD_DOSSIER.md sections 1-3 (positional, sizing, depth)
  - d2 Wilson 95% rubric (decision = half-width <= 5pp field / 15pp per-opp)
"""
import sqlite3, json, math, sys

DB = "data/opponent_db/opponent_db.sqlite"

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = (z/d) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (p, c-hw, c+hw, hw)

def grade(hw_pp, field=True):
    bar = 5.0 if field else 15.0
    return "decision" if hw_pp <= bar else "provisional"

c = sqlite3.connect(DB)

# raw-record sessions only (dossier convention)
RAW = [r[0] for r in c.execute("SELECT session_id FROM sessions WHERE has_raw_record=1")]
raw_set = set(RAW)
def inraw(sid): return sid in raw_set

# ----- load hands + actions for raw-record sessions -----
hands = {}
for r in c.execute("SELECT session_id,hand_idx,hero_seat,dealer_seat,sb_seat,bb_seat,"
                   "dealt_seats_json,start_stacks_json,bb,sb,ante,level FROM hands"):
    sid = r[0]
    if not inraw(sid): continue
    hands[(sid, r[1])] = dict(hero=r[2], dealer=r[3], sb=r[4], bb=r[5],
                              dealt=json.loads(r[6] or "[]"),
                              stacks=json.loads(r[7] or "{}"),
                              bb_chips=r[8], sb_chips=r[9], ante=r[10], level=r[11])

# voluntary preflop frame_diff actions per (sid,hand,seat) -> list of action dicts
acts_all = []
for r in c.execute("SELECT session_id,hand_idx,seat,is_hero,street,kind,amount_to,"
                   "amount_delta,stack_before,stack_depth_bb,all_in,voluntary,observed_via "
                   "FROM actions WHERE observed_via='frame_diff'"):
    sid = r[0]
    if not inraw(sid): continue
    acts_all.append(dict(sid=sid, hand=r[1], seat=r[2], is_hero=r[3], street=r[4],
                         kind=r[5], amount_to=r[6], amount_delta=r[7],
                         stack_before=r[8], depth=r[9], all_in=r[10], vol=r[11]))

# preflop non-hero voluntary actions
pf_opp = [a for a in acts_all if a["street"]==0 and not a["is_hero"] and a["vol"]==1]
# set of (hand_idx,seat) per session with >=1 such action
vol_keys = {}
for a in pf_opp:
    vol_keys.setdefault(a["sid"], set()).add((a["hand"], a["seat"]))

out = {}

# ================= T1: field VPIP (lower bound) =================
num = den = 0
for (sid, hidx), h in hands.items():
    vk = vol_keys.get(sid, set())
    for seat in h["dealt"]:
        if seat == h["hero"]: continue
        den += 1
        if (hidx, seat) in vk: num += 1
p, lo, hi, hw = wilson(num, den)
out["T1_field_vpip"] = dict(num=num, den=den, pct=round(p*100,1),
    ci=[round(lo*100,1),round(hi*100,1)], hw_pp=round(hw*100,1),
    grade=grade(hw*100), dossier="20.6% (372/1806) [B1]")

# ================= T2/T3: positional VPIP =================
# Position derivation per dossier 1: blinds = sb_seat/bb_seat, button=dealer_seat.
# non-blind non-button seats ordered first-to-act EP/MP/CO using seat order
# starting after BB (6-max table: order = SB,BB,EP,MP,CO,BTN going around).
# We order the dealt non-blind-non-button seats by position after BB.
def classify_positions(h):
    """Return dict seat->pos among dealt non-hero seats requires dealer+sb+bb."""
    dealer, sb, bb = h["dealer"], h["sb"], h["bb"]
    if dealer is None or sb is None or bb is None:
        return None
    dealt = sorted(h["dealt"])
    if not dealt: return None
    # ring order starting from SB: SB, BB, then around the table to BTN.
    # seats are 0..5; go around in increasing seat index modulo, starting after BB
    n = 6
    # build clockwise order of *dealt* seats starting at sb
    order = []
    s = sb
    for _ in range(n*2):
        if s in dealt and s not in order:
            order.append(s)
        s = (s+1) % n
        if len(order) == len(dealt): break
    pos = {}
    # order[0]=SB, order[1]=BB, order[-1] should be BTN(=dealer)
    if len(order) < 3:
        # heads-up-ish; classify what we can
        for i,seat in enumerate(order):
            if seat==sb: pos[seat]="SB"
            elif seat==bb: pos[seat]="BB"
            elif seat==dealer: pos[seat]="BTN"
        return pos
    for seat in order:
        if seat==sb: pos[seat]="SB"
        elif seat==bb: pos[seat]="BB"
        elif seat==dealer: pos[seat]="BTN"
    # remaining (non-blind non-button) ordered first-to-act after BB
    mid = [seat for seat in order if seat not in (sb,bb,dealer)]
    labels = ["EP","MP","CO"]
    # assign from the front (first to act = EP)
    for i,seat in enumerate(mid):
        if i < len(labels):
            pos[seat]=labels[i]
        else:
            pos[seat]="EP"  # extra early seats fold into EP bucket
    return pos

pos_num = {k:0 for k in ["EP","MP","CO","BTN","SB","BB"]}
pos_den = {k:0 for k in ["EP","MP","CO","BTN","SB","BB"]}
unclass = 0
for (sid,hidx), h in hands.items():
    vk = vol_keys.get(sid,set())
    pmap = classify_positions(h)
    for seat in h["dealt"]:
        if seat==h["hero"]: continue
        lab = pmap.get(seat) if pmap else None
        if lab is None:
            unclass += 1
            continue
        pos_den[lab]+=1
        if (hidx,seat) in vk: pos_num[lab]+=1

out["T2_positional"] = {}
for k in ["EP","MP","CO","BTN","SB","BB"]:
    p,lo,hi,hw = wilson(pos_num[k],pos_den[k])
    out["T2_positional"][k]=dict(num=pos_num[k],den=pos_den[k],pct=round(p*100,1),
        ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100))
out["T2_positional"]["unclassifiable"]=unclass

# ================= T4/T5: open sizing (first raise of hand) =================
# first raise of the hand = the earliest preflop raise/bet event across all
# (non-hero) seats in a hand. Dossier counts opponent first-raises.
# We reconstruct: for each hand, find the first preflop raise event (kind in
# raise/bet, voluntary) among non-hero seats; classify all-in vs xBB size.
first_raises = []  # dicts with all_in flag and xbb
for (sid,hidx),h in hands.items():
    bb_chips = h["bb_chips"] or 0
    if bb_chips<=0: continue
    # all preflop non-hero raise/bet events in this hand, ordered by amount_to
    cands = [a for a in pf_opp if a["sid"]==sid and a["hand"]==hidx
             and a["kind"] in ("raise","bet")]
    if not cands: continue
    # first raise = smallest amount_to (earliest escalation) — proxy for first
    cands.sort(key=lambda a:(a["amount_to"] if a["amount_to"] else 1e9))
    fr = cands[0]
    allin = bool(fr["all_in"])
    xbb = (fr["amount_to"]/bb_chips) if fr["amount_to"] else None
    first_raises.append(dict(allin=allin, xbb=xbb, amount_to=fr["amount_to"]))

n_fr = len(first_raises)
n_allin_fr = sum(1 for x in first_raises if x["allin"])
p,lo,hi,hw = wilson(n_allin_fr,n_fr)
out["T5_openjam_share"]=dict(num=n_allin_fr,den=n_fr,pct=round(p*100,1),
    ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100),
    dossier="23.0% (44/191) [B1]")

# non-allin open size bins (xbb), 0.5x bins as in dossier
nonallin = [x for x in first_raises if not x["allin"] and x["xbb"] is not None]
def binx(x):
    if x<=1.5: return "sub_min<=1.5"
    if x<=2.0: return "2.0x"
    if x<=2.5: return "2.5x"
    if x<=3.0: return "3.0x"
    if x<=3.5: return "3.5x"
    if x<=4.0: return "4.0x"
    if x<=5.0: return "5.0x"
    return "tail>5x"
from collections import Counter
bins = Counter(binx(x["xbb"]) for x in nonallin)
n_na = len(nonallin)
# 2x-mode dominance: <=2.0x share
n_le2 = sum(1 for x in nonallin if x["xbb"]<=2.0)
p,lo,hi,hw = wilson(n_le2,n_na)
out["T4_open_size_mix"]=dict(n_nonallin=n_na, bins=dict(bins),
    le2x_share=dict(num=n_le2,den=n_na,pct=round(p*100,1),
        ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100)),
    mode_3x_count=bins.get("3.0x",0),
    tail5plus_count=bins.get("5.0x",0)+bins.get("tail>5x",0),
    band_2_2_2_5_count=bins.get("2.5x",0),
    dossier="2.0x mode 55.1% (<=2.0x=81/147); 3.0x ~16%; 5x+ ~10%; 2.2-2.5x ~5%")

# ================= T6: limp share of preflop calls =================
# preflop opponent voluntary calls; limp = call to exactly 1bb (amount_to==bb)
calls = [a for a in pf_opp if a["kind"]=="call"]
n_calls = len(calls)
n_limps = 0
for a in calls:
    h = hands.get((a["sid"],a["hand"]))
    if not h: continue
    bb_chips = h["bb_chips"] or 0
    if bb_chips>0 and a["amount_to"] is not None and abs(a["amount_to"]-bb_chips)<1e-6:
        n_limps+=1
p,lo,hi,hw = wilson(n_limps,n_calls)
out["T6_limp_share"]=dict(num=n_limps,den=n_calls,pct=round(p*100,1),
    ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100),
    dossier="63% (95/150) [B1]")

# ================= T7: 5-15bb preflop jam rate per seat-hand =================
# per dossier 3b: denominator = non-hero dealt seat-hands with anchored pre-hand
# stack in band; numerator = seat-hands with >=1 observed preflop all-in.
# pre-hand depth = start_stacks_json[seat]/bb.
def seat_prehand_bb(h, seat):
    st = h["stacks"]
    bb_chips = h["bb_chips"] or 0
    v = st.get(str(seat), st.get(seat))
    if v is None or bb_chips<=0: return None
    return v/bb_chips

# preflop all-in seat-hands
pf_allin_keys = {}  # sid -> set((hand,seat))
for a in pf_opp:
    if a["all_in"]:
        pf_allin_keys.setdefault(a["sid"],set()).add((a["hand"],a["seat"]))

bands = [("0-5",0,5),("5-10",5,10),("10-15",10,15),("15-25",15,25),
         ("25-50",25,50),("50+",50,1e9)]
band_jam = {b[0]:[0,0] for b in bands}  # [num,den]
band_vpip = {b[0]:[0,0] for b in bands}
for (sid,hidx),h in hands.items():
    vk = vol_keys.get(sid,set())
    ak = pf_allin_keys.get(sid,set())
    for seat in h["dealt"]:
        if seat==h["hero"]: continue
        d = seat_prehand_bb(h,seat)
        if d is None: continue
        for name,lo_b,hi_b in bands:
            if lo_b<=d<hi_b:
                band_jam[name][1]+=1
                band_vpip[name][1]+=1
                if (hidx,seat) in ak: band_jam[name][0]+=1
                if (hidx,seat) in vk: band_vpip[name][0]+=1
                break

out["T7_jam_by_band"]={}
for name,_,_ in bands:
    num,den=band_jam[name]
    p,lo,hi,hw=wilson(num,den)
    out["T7_jam_by_band"][name]=dict(num=num,den=den,pct=round(p*100,1),
        ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100))
# combined 5-15
n515=band_jam["5-10"][0]+band_jam["10-15"][0]
d515=band_jam["5-10"][1]+band_jam["10-15"][1]
p,lo,hi,hw=wilson(n515,d515)
out["T7_jam_5_15_combined"]=dict(num=n515,den=d515,pct=round(p*100,1),
    ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100),
    dossier="8.5% (32/376) [B1] decision")
# combined 5-15 VPIP
n515v=band_vpip["5-10"][0]+band_vpip["10-15"][0]
d515v=band_vpip["5-10"][1]+band_vpip["10-15"][1]
p,lo,hi,hw=wilson(n515v,d515v)
out["T_vpip_5_15_combined"]=dict(num=n515v,den=d515v,pct=round(p*100,1),
    ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100),
    dossier="20.2% (76/376) [B1] decision")

# ================= T8: jam-regime break (per-ACTION all-in % by depth band) =================
# dossier 3a: opponent voluntary actions by actor stack_depth_bb at event time.
# denominator = all opponent voluntary frame_diff actions (any street) in band;
# numerator = those that are all_in.
act_bands = [("0-5",0,5),("5-10",5,10),("10-15",10,15),("15-25",15,25),
             ("25-50",25,50),("50+",50,1e9)]
ab = {b[0]:[0,0] for b in act_bands}
opp_vol_all = [a for a in acts_all if not a["is_hero"] and a["vol"]==1]
for a in opp_vol_all:
    d = a["depth"]
    if d is None: continue
    for name,lo_b,hi_b in act_bands:
        if lo_b<=d<hi_b:
            ab[name][1]+=1
            if a["all_in"]: ab[name][0]+=1
            break
out["T8_allin_by_actiondepth"]={}
for name,_,_ in act_bands:
    num,den=ab[name]
    p,lo,hi,hw=wilson(num,den)
    out["T8_allin_by_actiondepth"][name]=dict(num=num,den=den,pct=round(p*100,1),
        ci=[round(lo*100,1),round(hi*100,1)],hw_pp=round(hw*100,1),grade=grade(hw*100))

# ================= context counts =================
out["_meta"]=dict(
    raw_sessions=len(RAW),
    total_sessions=c.execute("SELECT count(*) FROM sessions").fetchone()[0],
    h4_counter=c.execute("SELECT count(*) FROM hands WHERE opp_observed=1").fetchone()[0],
    nonhero_dealt_seathands=sum(1 for (sid,hidx),h in hands.items()
        for seat in h["dealt"] if seat!=h["hero"]),
    opp_vol_frame_diff_actions=len(opp_vol_all),
    first_raises=n_fr,
)

print(json.dumps(out, indent=2))
with open("evals/h4_freeze_refresh_20260613/refreshed_constants.json","w") as f:
    json.dump(out,f,indent=2)
