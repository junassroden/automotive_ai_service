
import json
import threading
import urllib.request
import tkinter as tk
from tkinter import scrolledtext

API_URL = "http://127.0.0.1:8001/api/chat/"
HEALTH_URL = "http://127.0.0.1:8001/api/health/"

VEHICLE = {
    "vehicle_type": "SUV",
    "brand": "Toyota",
    "model": "Fortuner",
    "engine_type": "Diesel",
    "make_year": 2022,
    "region": "MIMAROPA",
    "mileage_range": "50000-75000",
    "mileage": 60000
}


class AutomotiveAIChat:
    def __init__(self, root):
        self.root = root
        self.root.title("Automotive AI Assistant")
        self.root.geometry("950x700")
        self.root.minsize(600, 450)
        self.root.configure(bg="#f3f4f6")
        self.busy = False
        self.placeholder = "Type your message..."

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(root, bg="#202938", padx=18, pady=12)
        header.grid(row=0, column=0, sticky="ew")

        tk.Label(
            header,
            text="Automotive AI Assistant",
            font=("Segoe UI", 17, "bold"),
            bg="#202938",
            fg="white"
        ).pack(anchor="w")

        tk.Label(
            header,
            text="Vehicle maintenance and fleet assistance",
            font=("Segoe UI", 10),
            bg="#202938",
            fg="#d1d5db"
        ).pack(anchor="w", pady=(3, 5))

        self.status = tk.Label(
            header,
            text="Connecting to AI service...",
            font=("Segoe UI", 9),
            bg="#202938",
            fg="#fbbf24"
        )
        self.status.pack(anchor="w")

        self.chat = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            state="disabled",
            font=("Segoe UI", 10),
            padx=14,
            pady=12,
            bg="white",
            fg="#111827",
            relief="flat",
            borderwidth=0,
            spacing3=7
        )
        self.chat.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=12,
            pady=(10, 5)
        )

        self.chat.tag_configure(
            "user",
            foreground="#1d4ed8",
            font=("Segoe UI", 10, "bold")
        )
        self.chat.tag_configure(
            "assistant",
            foreground="#047857",
            font=("Segoe UI", 10, "bold")
        )
        self.chat.tag_configure(
            "meta",
            foreground="#6b7280",
            font=("Segoe UI", 9)
        )
        self.chat.tag_configure(
            "error",
            foreground="#b91c1c"
        )

        bottom = tk.Frame(root, bg="#f3f4f6")
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        options = tk.Frame(bottom, bg="#f3f4f6", padx=12, pady=4)
        options.grid(row=0, column=0, sticky="ew")

        self.include_vehicle = tk.BooleanVar(value=True)

        tk.Checkbutton(
            options,
            text="Include test vehicle details",
            variable=self.include_vehicle,
            bg="#f3f4f6",
            font=("Segoe UI", 9)
        ).pack(side="left")

        tk.Button(
            options,
            text="Check Connection",
            command=self.check_connection,
            font=("Segoe UI", 9)
        ).pack(side="right")

        input_frame = tk.Frame(
            bottom,
            bg="white",
            padx=8,
            pady=8,
            highlightbackground="#cbd5e1",
            highlightthickness=1
        )
        input_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=12,
            pady=(3, 12)
        )
        input_frame.grid_columnconfigure(0, weight=1)

        self.message_input = tk.Entry(
            input_frame,
            font=("Segoe UI", 11),
            relief="flat",
            bd=0
        )
        self.message_input.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(8, 10),
            ipady=10
        )

        self.message_input.insert(0, self.placeholder)
        self.message_input.config(fg="#888888")
        self.message_input.bind("<FocusIn>", self.clear_placeholder)
        self.message_input.bind("<FocusOut>", self.restore_placeholder)
        self.message_input.bind("<Return>", self.send_message)

        self.send_button = tk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=22,
            pady=8,
            cursor="hand2"
        )
        self.send_button.grid(row=0, column=1, sticky="ns")

        self.add_message(
            "AI",
            "Hello! Ask me about vehicle maintenance, brakes, oil, "
            "cooling systems, safety, service schedules, or prediction scores."
        )

        self.root.after(300, self.check_connection)
        self.message_input.focus_set()

    def clear_placeholder(self, event=None):
        if self.message_input.get() == self.placeholder:
            self.message_input.delete(0, tk.END)
            self.message_input.config(fg="#111827")

    def restore_placeholder(self, event=None):
        if not self.message_input.get().strip():
            self.message_input.delete(0, tk.END)
            self.message_input.insert(0, self.placeholder)
            self.message_input.config(fg="#888888")

    def add_message(self, sender, message, meta=None, error=False):
        self.chat.config(state="normal")

        if sender == "You":
            tag = "user"
        elif error:
            tag = "error"
        else:
            tag = "assistant"

        self.chat.insert(tk.END, sender + "\n", tag)
        self.chat.insert(tk.END, str(message) + "\n")

        if meta:
            self.chat.insert(tk.END, str(meta) + "\n", "meta")

        self.chat.insert(tk.END, "\n")
        self.chat.config(state="disabled")
        self.chat.see(tk.END)

    def check_connection(self):
        self.status.config(
            text="Checking AI service...",
            fg="#fbbf24"
        )
        threading.Thread(
            target=self._check_connection,
            daemon=True
        ).start()

    def _check_connection(self):
        try:
            request = urllib.request.Request(HEALTH_URL)

            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))

            if data.get("status") == "ok":
                text = "Connected to AI service"
                color = "#86efac"
            else:
                text = "Unexpected AI service response"
                color = "#fca5a5"
        except Exception:
            text = "AI service offline. Start Django first."
            color = "#fca5a5"

        self.root.after(
            0,
            lambda: self.status.config(text=text, fg=color)
        )

    def send_message(self, event=None):
        message = self.message_input.get().strip()

        if not message or message == self.placeholder:
            return "break"

        if self.busy:
            return "break"

        self.message_input.delete(0, tk.END)
        self.message_input.config(fg="#111827")
        self.add_message("You", message)

        payload = {"message": message}

        if self.include_vehicle.get():
            payload["vehicle"] = VEHICLE

        self.busy = True
        self.send_button.config(state="disabled", text="Sending...")
        self.status.config(text="AI is responding...", fg="#fbbf24")

        threading.Thread(
            target=self._send_request,
            args=(payload,),
            daemon=True
        ).start()

        return "break"

    def _send_request(self, payload):
        try:
            body = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                API_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))

            self.root.after(
                0,
                lambda: self.display_result(result)
            )

        except Exception as error:
            details = str(error)
            self.root.after(
                0,
                lambda: self.finish_error(details)
            )

    def display_result(self, result):
        answer = result.get("answer", "No answer was returned.")
        intent = result.get("intent", "unknown")
        confidence = result.get("confidence")

        metadata = f"Intent: {intent}"

        if confidence is not None:
            try:
                metadata += f" | Confidence: {float(confidence):.4f}"
            except (TypeError, ValueError):
                pass

        self.add_message("AI", answer, metadata)

        data = result.get("data")

        if data:
            self.add_message(
                "AI details",
                json.dumps(data, indent=2, ensure_ascii=False)
            )

        self.finish_request()

    def finish_error(self, message):
        self.add_message(
            "Connection error",
            message,
            error=True
        )
        self.finish_request()

    def finish_request(self):
        self.busy = False
        self.send_button.config(state="normal", text="Send")
        self.status.config(text="Ready", fg="#86efac")
        self.message_input.focus_set()


if __name__ == "__main__":
    root = tk.Tk()
    app = AutomotiveAIChat(root)
    root.mainloop()
