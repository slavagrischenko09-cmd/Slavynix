import tkinter as tk

from gui import VoiceSwapGUI


def main() -> None:
    root = tk.Tk()
    app = VoiceSwapGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
