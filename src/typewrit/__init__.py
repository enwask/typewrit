from typewrit.llm.completion import get_completions


if __name__ == "typewrit":
    def main() -> None:
        while True:
            inp = input("> ")
            if inp.lower() in {'exit', 'quit', 'q'}:
                break
            print(list(map(str, get_completions(inp))))
