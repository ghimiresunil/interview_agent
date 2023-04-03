from helper import FuseBot

if __name__ == "__main__":
    fb = FuseBot()
    hard_skills=['pandas','docker','github']
    while(True):
        inp = input("Enter User Text: ")
        res = fb.get_answer_in_text(inp,hard_skills)
        print("RES:", res)