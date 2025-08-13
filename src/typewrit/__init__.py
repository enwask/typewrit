from typewrit.llm.completion import get_completions


if __name__ == "typewrit":
    def main() -> None:
        while True:
            inp = input("> ")
            if inp.lower() in {'exit', 'quit', 'q'}:
                break
            print(get_completions(inp))
