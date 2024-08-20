import csv
import sys
from util import Node, QueueFrontier

names = {}
people = {}
movies = {}


def load_data(dir):
    with open(f"{dir}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    with open(f"{dir}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    with open(f"{dir}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [dir]")
    dir = sys.argv[1] if len(sys.argv) == 2 else "large"

    print("Data: loading")
    load_data(dir)
    print("Data: loaded")

    source = person_id_for_name(input("Name: "))
    if source is None:
        sys.exit("Person not found.")
    target = person_id_for_name(input("Name: "))
    if target is None:
        sys.exit("Person not found.")

    path = shortest_path(source, target)

    if path is None:
        print("Not connected!")
    else:
        degrees = len(path)
        print(f"{degrees} degrees of separation.")
        path = [(None, source)] + path
        for i in range(degrees):
            person1 = people[path[i][1]]["name"]
            person2 = people[path[i + 1][1]]["name"]
            movie = movies[path[i + 1][0]]["title"]
            print(f"{i + 1}: {person1} and {person2} starred in {movie}")


def shortest_path(source, target):
    start = Node(source, None, None)
    frontier = QueueFrontier()
    frontier.add(start)
    explored = set()

    while True:
        if len(explored) % 100 == 0:
            print('Actors explored: ', len(explored))
            print('Nodes left: ', len(frontier.frontier))

        if frontier.empty():
            print('Frontier is empty: No connection between actors')
            print(len(explored), 'No solution found')
            return None

        curr_node = frontier.remove()
        explored.add(curr_node.state)

        for action, state in neighbors_for_person(curr_node.state):

            if state == target:
                print('Solution Found')
                print(len(explored), 'Find solution')
                path = []
                path.append((action, state))

                while curr_node.parent != None:
                    path.append((curr_node.action, curr_node.state))
                    curr_node = curr_node.parent
                path.reverse()
                return path

            if not frontier.contains_state(state) and state not in explored:
                new_node = Node(state, curr_node, action)
                frontier.add(new_node)


def person_id_for_name(name):
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended person id: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors


if __name__ == "__main__":
    main()