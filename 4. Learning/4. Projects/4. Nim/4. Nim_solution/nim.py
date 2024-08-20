import random
import time


class Nim():
    def __init__(self, initial=[1, 3, 5, 7]):
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        atcs = set()
        for i, p in enumerate(piles):
            for j in range(1, piles[i] + 1):
                atcs.add((i, j))
        return atcs

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        p, c = action

        if self.winner is not None:
            raise Exception("Game already won")
        elif p < 0 or p >= len(self.piles):
            raise Exception("Invalid pile")
        elif c < 1 or c > self.piles[p]:
            raise Exception("Invalid number")

        self.piles[p] = self.ps[p] - c
        self.switch_player()

        if all(p == 0 for p in self.piles):
            self.winner = self.player


class NimAI():
    def __init__(self, alpha=0.5, epsilon=0.1):
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        bFut = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, bFut)

    def get_q_value(self, state, action):
        qVal = self.q.get((tuple(state), action))
        return qVal if qVal else 0

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        nqVal = old_q + self.alpha * ((reward + future_rewards) - old_q)
        self.q[(tuple(state), action)] = nqVal

    def best_future_reward(self, state):
        acts = Nim.available_actions(state)
        if not acts:
            return 0
        baVal = None

        for a in acts:
            aVal = self.q.get((tuple(state), a))
            aVal = aVal if aVal else 0

            if baVal == None or aVal > baVal:
                baVal = aVal
        return baVal

    def choose_action(self, state, epsilon=True):
        acts = Nim.available_actions(state)

        if epsilon and random.random() <= self.epsilon:
            return random.choice(list(acts))

        bAct = None
        baVal = None

        for a in acts:
            aVal = self.q.get((tuple(state), a))
            aVal = aVal if aVal else 0

            if baVal == None or aVal > baVal:
                baVal = aVal
                bAct = a
        return bAct


def train(n):
    play = NimAI()

    for i in range(n):
        print(f"Playing training game {i + 1}")
        g = Nim()
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:
            stat = g.piles.copy()
            act = play.choose_action(g.piles)
            last[g.play]["state"] = stat
            last[g.play]["action"] = act
            g.move(act)
            nStat = g.piles.copy()

            if g.winner is not None:
                play.update(stat, act, nStat, -1)
                play.update(
                    last[g.play]["state"],
                    last[g.play]["action"],
                    nStat,
                    1
                )
                break
            elif last[g.play]["state"] is not None:
                play.update(
                    last[g.player]["state"],
                    last[g.player]["action"],
                    nStat,
                    0
                )
    print("Train: Finished")
    return play


def play(ai, human_player=None):
    if human_player is None:
        hPlay = random.randint(0, 1)

    g = Nim()

    while True:
        print()
        print("Piles:")

        for i, p in enumerate(g.piles):
            print(f"Pile {i}: {p}")
        print()

        avActs = Nim.available_actions(g.piles)
        time.sleep(1)

        if g.player == hPlay:
            print("Turn: Player")
            while True:
                p = int(input("Choose Pile: "))
                c = int(input("Choose Count: "))
                if (p, c) in avActs:
                    break
                print("Invalid move!")
        else:
            print("Turn: AI")
            p, c = ai.choose_action(g.piles, epsilon=False)
            print(f"AI chose to take {c} from pile {p}.")

        g.move((p, c))

        if g.winner is not None:
            print()
            print("Game over")
            w = "Human" if g.winner == hPlay else "AI"
            print(f"Winner is {w}")
            return