 
from tkinter import *
from PIL import Image, ImageTk

def start():
        app = Tk()
        app.geometry("400x400")


        def get_x_and_y(event):
            global lasx, lasy
            lasx, lasy = event.x, event.y

        def draw_smth(event):
            global lasx, lasy
            canvas.create_line((lasx, lasy, event.x, event.y), fill='black', width=10)
            lasx, lasy = event.x, event.y

        def save():
            canvas.postscript(file="./user_generated_picture/circles.eps")
            img = Image.open("./user_generated_picture/circles.eps")
            img.save("user_number.png", "png")

            app.destroy()


        canvas = Canvas(app, bg='black')
        canvas.pack(anchor='nw', fill='both', expand=1)

        canvas.bind("<Button-1>", get_x_and_y)
        canvas.bind("<B1-Motion>", draw_smth)


        image = Image.open("blank.png")
        image = image.resize((400,400), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0,0, image=image, anchor='nw')

        button = Button(app, text="Save", command=save)
        button.pack(side="bottom")


        app.mainloop()

if __name__ == '__main__':
    start()

