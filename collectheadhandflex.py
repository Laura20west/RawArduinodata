import pygame
import serial
import csv
import string
from collections import deque
import os

# =====================================
# SERIAL SETTINGS
# =====================================

SERIAL_PORT = "COM8"
BAUD_RATE = 115200

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

pygame.time.wait(2000)

# =====================================
# DATASET SETTINGS
# =====================================

WINDOW_SIZE = 60       # Frames per sample
FEATURES = 17          # Total sensor features
SAMPLES_PER_LETTER = 100

DATASET_FILE = "dataset_mpuallalpha.csv"

# =====================================
# RECORDING STATE
# =====================================

recording = False

# Stores the current 60-frame sequence
sample_buffer = []

# Current letter being recorded
current_label = "A"

# Number of complete samples recorded
samples_for_current_label = 0

# Total samples in dataset
total_saved = 0

# =====================================
# PYGAME
# =====================================

pygame.init()

WIDTH = 1200
HEIGHT = 700

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT)
)

pygame.display.set_caption(
    "NSL Gesture Data Collector - Automatic 100 Samples"
)

font = pygame.font.SysFont(
    None,
    36
)

big_font = pygame.font.SysFont(
    None,
    50
)

small_font = pygame.font.SysFont(
    None,
    28
)

clock = pygame.time.Clock()

# =====================================
# SMOOTHING
# =====================================

histories = [
    deque(maxlen=5)
    for _ in range(FEATURES)
]


def smooth(index, value):

    histories[index].append(value)

    return (
        sum(histories[index])
        / len(histories[index])
    )


# =====================================
# SAVE SAMPLE
# =====================================

def save_gesture(label, sequence):

    row = [label]

    for frame in sequence:
        row.extend(frame)

    with open(
        DATASET_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(row)


# =====================================
# DATASET COUNT
# =====================================

def count_samples():

    if not os.path.exists(DATASET_FILE):
        return 0

    with open(
        DATASET_FILE,
        "r",
        newline=""
    ) as f:

        return sum(
            1 for _ in f
        )


# =====================================
# INITIAL COUNT
# =====================================

total_saved = count_samples()


# =====================================
# MAIN LOOP
# =====================================

running = True

latest_values = [0] * FEATURES

while running:

    screen.fill(
        (20, 20, 25)
    )

    # =================================
    # EVENTS
    # =================================

    for event in pygame.event.get():

        # -------------------------------
        # CLOSE WINDOW
        # -------------------------------

        if event.type == pygame.QUIT:

            running = False


        # -------------------------------
        # KEYBOARD
        # -------------------------------

        if event.type == pygame.KEYDOWN:

            # =================================
            # SELECT LETTER
            # =================================

            if (
                event.unicode
                and event.unicode.upper()
                in string.ascii_uppercase
            ):

                selected_label = (
                    event.unicode.upper()
                )

                # Only allow a new label
                # when NOT currently recording
                if not recording:

                    current_label = selected_label

                    # Reset counters
                    samples_for_current_label = 0
                    sample_buffer = []

                    # Automatically start recording
                    recording = True

                    print(
                        f"\n================================="
                    )

                    print(
                        f"STARTING RECORDING: {current_label}"
                    )

                    print(
                        f"Target: {SAMPLES_PER_LETTER} samples"
                    )

                    print(
                        f"Each sample: {WINDOW_SIZE} frames"
                    )

                    print(
                        f"=================================\n"
                    )

                else:

                    print(
                        f"Currently recording "
                        f"{current_label}. "
                        f"Please wait until all "
                        f"{SAMPLES_PER_LETTER} samples are complete."
                    )


            # =================================
            # ESCAPE TO STOP RECORDING
            # =================================

            if event.key == pygame.K_ESCAPE:

                if recording:

                    recording = False

                    sample_buffer = []

                    print(
                        f"\nRecording of {current_label} "
                        f"cancelled."
                    )

                    print(
                        f"Completed: "
                        f"{samples_for_current_label}/"
                        f"{SAMPLES_PER_LETTER}"
                    )

                    print(
                        "Press another letter to begin.\n"
                    )


    # =====================================
    # SERIAL READ
    # =====================================

    try:

        raw = arduino.readline()

        if raw:

            line = raw.decode(
                "ascii",
                errors="ignore"
            ).strip()

            parts = line.split(",")

            # Make sure we received exactly
            # 17 sensor values
            if len(parts) == FEATURES:

                try:

                    values = list(
                        map(
                            float,
                            parts
                        )
                    )

                except ValueError:

                    continue


                # =================================
                # SMOOTH SENSOR DATA
                # =================================

                smoothed = []

                for i in range(FEATURES):

                    smoothed.append(
                        smooth(
                            i,
                            values[i]
                        )
                    )

                latest_values = smoothed


                # =================================
                # RECORDING
                # =================================

                if recording:

                    # Add current frame
                    sample_buffer.append(
                        smoothed.copy()
                    )


                    # =================================
                    # 60 FRAMES COMPLETE
                    # =================================

                    if len(sample_buffer) >= WINDOW_SIZE:

                        # Save one complete sample
                        save_gesture(
                            current_label,
                            sample_buffer
                        )

                        # Update counters
                        samples_for_current_label += 1
                        total_saved += 1

                        print(
                            f"Saved sample "
                            f"{samples_for_current_label}/"
                            f"{SAMPLES_PER_LETTER} "
                            f"for {current_label}"
                        )

                        print(
                            f"Total dataset samples: "
                            f"{total_saved}"
                        )


                        # Reset frame buffer
                        # for the next sample
                        sample_buffer = []


                        # =================================
                        # 100 SAMPLES COMPLETE
                        # =================================

                        if (
                            samples_for_current_label
                            >= SAMPLES_PER_LETTER
                        ):

                            recording = False

                            print(
                                f"\n================================="
                            )

                            print(
                                f"COMPLETED: {current_label}"
                            )

                            print(
                                f"{SAMPLES_PER_LETTER} samples saved."
                            )

                            print(
                                "READY FOR NEXT LETTER."
                            )

                            print(
                                f"Press another letter "
                                f"A-Z to continue."
                            )

                            print(
                                f"=================================\n"
                            )


    except Exception as e:

        print(
            "ERROR:",
            e
        )


    # =====================================
    # HUD
    # =====================================

    y = 20

    # -------------------------------------
    # TITLE
    # -------------------------------------

    screen.blit(

        big_font.render(
            "NSL DATA COLLECTOR",
            True,
            (255, 255, 255)
        ),

        (20, y)
    )

    y += 70


    # -------------------------------------
    # CURRENT LABEL
    # -------------------------------------

    screen.blit(

        font.render(
            f"Current Label : {current_label}",
            True,
            (255, 255, 255)
        ),

        (20, y)
    )

    y += 45


    # -------------------------------------
    # RECORDING STATUS
    # -------------------------------------

    if recording:

        status_text = "RECORDING"

        status_color = (
            255,
            80,
            80
        )

    else:

        status_text = "READY"

        status_color = (
            0,
            255,
            100
        )


    screen.blit(

        font.render(
            f"Status : {status_text}",
            True,
            status_color
        ),

        (20, y)
    )

    y += 45


    # -------------------------------------
    # SAMPLE COUNT
    # -------------------------------------

    screen.blit(

        font.render(
            f"Samples for {current_label}: "
            f"{samples_for_current_label}/"
            f"{SAMPLES_PER_LETTER}",
            True,
            (255, 255, 255)
        ),

        (20, y)
    )

    y += 45


    # -------------------------------------
    # FRAME COUNT
    # -------------------------------------

    screen.blit(

        font.render(
            f"Current Sample Frames: "
            f"{len(sample_buffer)}/"
            f"{WINDOW_SIZE}",
            True,
            (0, 255, 0)
        ),

        (20, y)
    )

    y += 45


    # -------------------------------------
    # TOTAL DATASET
    # -------------------------------------

    screen.blit(

        font.render(
            f"Total Samples Saved: "
            f"{total_saved}",
            True,
            (0, 255, 255)
        ),

        (20, y)
    )

    y += 55


    # -------------------------------------
    # INSTRUCTIONS
    # -------------------------------------

    if recording:

        instruction = (
            f"Recording {current_label}... "
            f"Please hold the gesture."
        )

    else:

        instruction = (
            "Press A-Z to automatically "
            "record 100 samples."
        )


    screen.blit(

        small_font.render(
            instruction,
            True,
            (255, 255, 255)
        ),

        (20, y)
    )

    y += 35


    screen.blit(

        small_font.render(
            "Press ESC to cancel current recording.",
            True,
            (200, 200, 200)
        ),

        (20, y)
    )


    # =====================================
    # SENSOR VALUES
    # =====================================

    names = [

        "Thumb",
        "Index",
        "Middle",
        "Ring",
        "Pinky",

        "Hand AccX",
        "Hand AccY",
        "Hand AccZ",

        "Hand GyroX",
        "Hand GyroY",
        "Hand GyroZ",

        "Head AccX",
        "Head AccY",
        "Head AccZ",

        "Head GyroX",
        "Head GyroY",
        "Head GyroZ"
    ]


    start_y = 120
    start_x = 600

    for i in range(FEATURES):

        screen.blit(

            small_font.render(
                f"{names[i]} : "
                f"{latest_values[i]:.2f}",
                True,
                (255, 255, 255)
            ),

            (
                start_x,
                start_y + i * 32
            )
        )


    # =====================================
    # SAMPLE PROGRESS BAR
    # =====================================

    # Progress through the current
    # 60-frame sample

    frame_progress = (
        len(sample_buffer)
        / WINDOW_SIZE
    )

    pygame.draw.rect(

        screen,

        (50, 50, 50),

        (
            20,
            550,
            500,
            30
        )
    )


    pygame.draw.rect(

        screen,

        (0, 255, 0),

        (
            20,
            550,
            int(
                500
                * frame_progress
            ),
            30
        )
    )


    # =====================================
    # 100-SAMPLE PROGRESS BAR
    # =====================================

    sample_progress = (
        samples_for_current_label
        / SAMPLES_PER_LETTER
    )

    pygame.draw.rect(

        screen,

        (50, 50, 50),

        (
            20,
            600,
            500,
            30
        )
    )


    pygame.draw.rect(

        screen,

        (0, 200, 255),

        (
            20,
            600,
            int(
                500
                * sample_progress
            ),
            30
        )
    )


    # =====================================
    # UPDATE DISPLAY
    # =====================================

    pygame.display.flip()

    clock.tick(60)


# =====================================
# CLEANUP
# =====================================

arduino.close()

pygame.quit()

