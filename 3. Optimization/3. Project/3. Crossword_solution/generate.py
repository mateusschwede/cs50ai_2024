import sys
from collections import deque
from crossword import *


class CrosswordCreator():
    def __init__(self, crossword):
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        listLetters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        
        for var, w in assignment.items():
            dir = var.direction
            for k in range(len(w)):
                i = var.i + (k if dir == Variable.DOWN else 0)
                j = var.j + (k if dir == Variable.ACROSS else 0)
                listLetters[i][j] = w[k]
        return listLetters

    def print(self, assignment):
        listLetters = self.letter_grid(assignment)
        
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(listLetters[i][j] or " ", end="")
                else:
                    print(" ", end="")
            print()

    def save(self, assignment, filename):
        from PIL import Image, ImageDraw, ImageFont
        
        cellSize = 100
        cellBorder = 2
        intSize = cellSize - 2 * cellBorder
        listLetters = self.letter_grid(assignment)
        figure = Image.new("RGBA", (self.crossword.width * cellSize, self.crossword.height * cellSize), "black")
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(figure)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                rect = [
                    (j * cellSize + cellBorder, i * cellSize + cellBorder),
                    ((j + 1) * cellSize - cellBorder, (i + 1) * cellSize - cellBorder)
                ]
                
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if listLetters[i][j]:
                        w, h = draw.textsize(listLetters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((intSize - w) / 2),
                             rect[0][1] + ((intSize - h) / 2) - 10),
                            listLetters[i][j], fill="black", font=font
                        )
        figure.save(filename)

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        for var, listWords in self.domains.items():
            remWords = set()
            for w in listWords:
                if len(w) != var.length:
                    remWords.add(w)
            self.domains[var] = listWords.difference(remWords)

    def revise(self, x, y):
        rev = False
        overlap = self.crossword.overlaps[x, y]

        if overlap:
            v1, v2 = overlap
            remXS = set()
            for xi in self.domains[x]:
                overlaps = False
                for yj in self.domains[y]:
                    if xi != yj and xi[v1] == yj[v2]:
                        overlaps = True
                        break
                if not overlaps:
                    remXS.add(xi)
            if remXS:
                self.domains[x] = self.domains[x].difference(remXS)
                rev = True
        return rev

    def ac3(self, arcs=None):
        if arcs is None:
            arcs = deque()
            for v1 in self.crossword.variables:
                for v2 in self.crossword.neighbors(v1):
                    arcs.appendleft((v1, v2))
        else:
            arcs = deque(arcs)

        while arcs:
            x, y = arcs.pop()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                for z in self.crossword.neighbors(x) - {y}:
                    arcs.appendleft((z, x))
        return True

    def assignment_complete(self, assignment):
        for var in self.crossword.variables:
            if var not in assignment.keys():
                return False

            if assignment[var] not in self.crossword.words:
                return False
        return True

    def consistent(self, assignment):
        for varX, wordX in assignment.items():
            if varX.length != len(wordX):
                return False

            for varY, wordY in assignment.items():
                if varX != varY:                  
                    if wordX == wordY:
                        return False

                    overlap = self.crossword.overlaps[varX, varY]
                    if overlap:
                        a, b = overlap
                        if wordX[a] != wordY[b]:
                            return False
        return True

    def order_domain_values(self, var, assignment):
        neighB = self.crossword.neighbors(var)
      
        for v in assignment:
            if v in neighB:
                neighB.remove(v)

        res = []
        for v in self.domains[var]:
            ruledOut = 0
            for neigh in neighB:
                for v2 in self.domains[neigh]:
                    overlap = self.crossword.overlaps[var, neigh]
                    if overlap:
                        a, b = overlap
                        if v[a] != v2[b]:
                            ruledOut = ruledOut + 1          
            res.append([v, ruledOut])

        res.sort(key=lambda x: x[1])
        return [i[0] for i in res]

    def select_unassigned_variable(self, assignment):
        potVar = []
        for v in self.crossword.variables:
            if v not in assignment:
                potVar.append([v, len(self.domains[v]), len(self.crossword.neighbors(v))])

        if potVar:
            potVar.sort(key=lambda x: (x[1], -x[2]))
            return potVar[0][0]
        return None

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment
        v = self.select_unassigned_variable(assignment)

        for value in self.order_domain_values(v, assignment):
            assignment[v] = value

            if self.consistent(assignment):
                res = self.backtrack(assignment)
                if res:
                    return res
                assignment.pop(v, None)
        return None


def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Execute: python generate.py structure words [output]")

    struct = sys.argv[1]
    listWords = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) == 4 else None
    crossWrd = Crossword(struct, listWords)
    cre = CrosswordCreator(crossWrd)
    assign = cre.solve()

    if assign is None:
        print("None solution!")
    else:
        cre.print(assign)
        if out:
            cre.save(assign, out)


if __name__ == "__main__":
    main()