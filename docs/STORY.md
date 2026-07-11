# The Story: teaching a computer to see a ping-pong paddle

*Written to be understood by someone who has never touched a line of code or heard the
words "neural network." No jargon survives more than one sentence without being explained.*

---

## What we set out to do

I wanted a computer to look through a camera and point at a ping-pong paddle — draw a
box around it, live, as the paddle moves. Simple to say, and something a two-year-old
does effortlessly. For a computer it is genuinely hard, and the *way* it's hard is the
interesting part.

There was a second goal hiding behind the first. The computer I wanted this to run on
isn't a big desktop. It's a small **edge board** from Qualcomm — the kind of chip that
goes inside a phone, a doorbell camera, or a car. These chips have a special piece of
silicon just for AI, called an **NPU** (Neural Processing Unit). I wanted to actually
use it. So the real journey was: build something that works, then make it run fast on a
tiny piece of hardware.

---

## Part 1 — Where does a computer learn "paddle" from?

A computer doesn't know what a paddle is. You have to *show* it — thousands of times —
and each time also tell it "the paddle is right **here**." That collection of
example-photos-with-answers is called a **dataset**, and it is the single most
important ingredient. Everything downstream depends on it.

So I recorded video of ping-pong paddles: on tables, in hands, near and far, in
different rooms. Then, on each frame, I drew a rectangle around the paddle by hand.
That rectangle is the **label** — the answer key. A photo with its box is one
**example**. I collected a few hundred of them (about 670 in total), and I also kept
some photos with **no paddle at all** — those teach the computer what "nothing here"
looks like, which matters just as much.

> **The one thing to remember from Part 1:** an AI is only as good as the examples you
> show it. More varied examples beat a cleverer program almost every time. This becomes
> the moral of the whole story.

---

## Part 2 — Building a brain from scratch (and why it half-failed)

For the first attempt I built a small artificial "brain" — a **neural network** — from
nothing, and trained it only on my paddle photos.

Think of a neural network as a machine with millions of tiny adjustable knobs. At the
start the knobs are random, so it guesses nonsense. **Training** means: show it a photo,
let it guess where the paddle is, compare its guess to my hand-drawn box, and nudge
every knob a hair in the direction that would have made the guess better. Do that for
every photo, over and over, for hours. Slowly the guesses stop being nonsense. On the
Mac, using its graphics chip to do the math, this took a couple of hours.

**It worked — on paper.** On photos that looked like my training photos (a person
standing back, the whole scene visible, decent light) it found the paddle nicely.

Then I pointed a real webcam at myself and it went **blind**. When I leaned in close,
my face filling the frame, in a slightly dim room — nothing. It couldn't find a paddle
it would have spotted easily from across the room.

I didn't want to guess why, so I tested it like a doctor runs tests:

- Made the paddle small, then big → no difference. **Not about size.**
- Stretched the image to a different shape → no difference. **Not about the frame shape.**
- Cropped my big face out of the picture → *suddenly it found the paddle.* **It was the
  face.**
- Brightened the dim image → *the paddle score jumped.* **The darkness hurt too.**

The diagnosis: my little brain had only ever seen photos of *people standing far back in
well-lit rooms*, because that's what my few hundred training photos looked like. A
**giant face in a dark close-up** was a situation it had literally never encountered. It
hadn't learned "paddle" — it had **memorized my photo album**. Show it anything outside
that album and it panics.

> **The one thing to remember from Part 2:** a small brain trained on a small,
> similar-looking pile of photos doesn't *understand* — it memorizes. It looks smart
> until the world hands it something new.

---

## Part 3 — Standing on a giant's shoulders (this is the fix)

There's a much better idea than building a brain from scratch: **start with a brain that
has already seen the world.**

Researchers have trained enormous networks on *millions* of everyday photos — people,
dogs, cars, chairs, faces, in every lighting and every distance you can imagine. One
famous, freely available family of these is called **YOLO** (it stands for "You Only
Look Once," because it spots everything in a picture in a single glance). The version I
used, **YOLOv8-nano**, is the *smallest* one — deliberately, because it has to fit on a
tiny chip later.

This pre-trained YOLO already knows, deep down, what edges and shapes and faces and
"objects held in a hand" look like. It just doesn't know the specific word "paddle." So
instead of teaching it everything from zero, I only had to teach it that **last little
bit**. This is called **fine-tuning** — like hiring someone who already speaks five
languages and just teaching them your company's jargon, rather than raising a child from
birth. The training was quick and used the exact same few hundred photos as before.

The difference was night and day. The fine-tuned YOLO found the paddle **everywhere** —
including the close-up-face-in-a-dark-room shot that had completely blinded my
from-scratch brain. Same photos, same computer, same amount of my effort. The only thing
that changed was **starting from a model that had already seen the world.**

To put a number on "how good": the from-scratch brain scored about **0.56** on our
accuracy measure; the fine-tuned YOLO scored about **0.98** (where 1.0 is perfect). But
the number that mattered most was simply: *it works in my actual room now.*

> **The one thing to remember from Part 3:** don't build from zero if a giant has
> already done 99% of the work for free. Fine-tuning a big pre-trained model beat my
> hand-built one, easily.

---

## Part 4 — Seeing it live in a browser

A score in a spreadsheet is abstract. I wanted to *watch* it work. So I wrote a tiny
program that opens the webcam, runs each frame through the model, draws the green box,
and shows the result in a web browser — like a security-camera feed with a box painted
on the paddle. You open a page, pick your camera, hit **start**, and there it is: a live
video with the paddle boxed as you wave it around.

The clever trick underneath: the program that *shows* the video doesn't care *which*
brain is doing the looking. From-scratch brain, YOLO on the regular processor, or YOLO
on the AI chip — I could swap the engine and the video feed never changed a line. That
one design choice made everything afterward easy.

---

## Part 5 — Moving onto the little Qualcomm board

Everything so far ran on my Mac. The real target was the small **Qualcomm board**. The
rule I stuck to throughout: **train on the big computer, run on the little one.**

Moving the model over means "freezing" it into a portable file format (called **ONNX**)
that the board can read without needing all the heavy training software. I copied that
file to the board and ran it on the board's ordinary processor (its **CPU**). It worked
— about 24 frames per second, smooth enough for live video. 

But the board has that special AI chip, the **NPU**, sitting right there unused. That's
what it's *for*. So the next step was to get the model onto it.

---

## Part 6 — The AI chip, and the surprise that taught me the most<a name="part-6"></a>

Getting a model onto Qualcomm's NPU is not a copy-paste job. The chip doesn't run the
model's math the same careful way a regular processor does. To go fast and sip power, it
uses **simpler, rougher numbers** — instead of long precise decimals, it rounds
everything to small whole numbers. This rounding-down is called **quantization**, and
it's how these chips achieve their speed.

Here I hit a trap that cost real detective work. When I first rounded *everything* down
the same way, the model's **confidence score collapsed to zero** — it technically found
the paddle but reported "0% sure," which is useless. The reason: the score is a tiny
number between 0 and 1, but the box coordinates are big numbers like 400. Forcing both
through the *same* coarse rounding crushed the delicate little score into nothing. The
fix was to let the chip keep **more precision for the in-between calculations** (a mode
called "A16W8") — enough to protect the fragile score. After that, the NPU reported a
healthy confident detection again.

And now the headline: **the NPU does the model's raw math about 84 times faster than
the CPU** — 1.7 thousandths of a second versus 145. Astonishing.

So I expected the live video to fly. I measured the *whole* pipeline honestly, and at
first it *didn't*: the NPU version came out slightly slower overall — around 57
milliseconds per frame versus 42 on the CPU. The 84×-faster engine was losing the race.

That made no sense, so I went looking. The math was never the slow part — the chip did
its 1.7 ms and then sat waiting. The waiting had a cause, and it wasn't what I assumed.
To go fast, the NPU wants its numbers in a compact 16-bit whole-number format, not the
long decimals the camera frame arrives as. Something has to translate every frame into
that format, and the answer back out of it. I had been letting the chip's toolkit do that
translation — and it did it the slow way, **one number at a time**, three hundred thousand
numbers per frame, on the CPU. *That* was the 30 lost milliseconds. (I'd first guessed it
was the cost of writing files to storage — but the files live in RAM, so that part was
nearly free. Measuring, again, corrected my guess.)

The fix was to do the translation the smart way: convert the *whole* frame at once with a
fast math library (a fraction of a millisecond), hand the chip exactly the bytes it wants,
and read its answer back the same way. No more translating number-by-number.

With that fixed, I measured again — and the NPU **won, cleanly: about 25 milliseconds per
frame versus 42 on the CPU, a real 1.7× speedup end-to-end**, with the identical detection
(same green box, same confidence). The blazing engine finally got a road to match.

This detour is one of the most useful lessons in all of engineering, and I got to
*measure* every step of it rather than just read about it:

> **The one thing to remember from Part 6:** a faster engine only helps if the road to it
> isn't the bottleneck — and the bottleneck is rarely where you first guess. The NPU's raw
> math was 84× faster all along; the win only showed up once I stopped feeding it in the
> wrong format and translating one number at a time. Feed the accelerator the way it wants
> to be fed, and it wins.

And I got what I came for: the paddle recognized **live, on the Qualcomm NPU**, a green
box tracking it in real time, *faster* than the CPU — the exact thing the from-scratch
brain could never do.

---

## Am I forgetting anything? (the honest checklist)

The original plan was: dataset → training → run on Mac → run on the board's CPU →
convert for Qualcomm → run on the board's NPU. We did all of that. Along the way, these
pieces turned out to matter just as much, and they're all in this repo:

- **The live web demo** (Part 4) — proof it works on moving video, not just still photos.
- **The diagnosis of *why* the first model failed** (Part 2) — the face, the darkness.
  Knowing *why* something breaks is worth more than the fix itself.
- **Measuring, not assuming** — the CPU-vs-NPU benchmark (Part 6) is the whole reason we
  learned the "road to the chip" lesson. The NPU first *looked* slower; only measuring —
  and then measuring again after the fix — revealed both the problem and the real 1.7× win.
- **The quantization trap** (Part 6) — the collapsing-score bug and its fix.
- **A few real-world gremlins** — a USB camera that needed a full reboot (not a replug)
  to come back, a camera that stayed "locked" if you forgot to press stop. The
  unglamorous stuff that's 90% of making hardware actually work.

---

## The four lessons, in one breath

1. **Your data is your ceiling.** A few hundred similar photos → a model that memorizes,
   not one that understands.
2. **Don't build from zero.** Fine-tuning a giant pre-trained model beat my hand-built
   one without contest.
3. **Measure the whole thing, honestly.** The "84× faster" chip *looked* like it lost the
   real race — until measuring showed the loss was a fixable format-conversion cost, not
   the chip.
4. **The bottleneck is rarely where you think.** It wasn't the math, and it wasn't the disk
   — it was translating each frame into the chip's format one number at a time. Fix the
   road, and the fast engine finally wins.

If you want to do all of this yourself, every command is in
**[REPRODUCE.md](REPRODUCE.md)**.
