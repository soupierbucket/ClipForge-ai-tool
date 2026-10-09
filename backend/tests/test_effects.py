import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.services.effect_planner import EffectItem, build_video_filter, validate_effects_plan
from app.utils.ffmpeg import probe_video_dimensions

class EffectsPlanTests(unittest.TestCase):
    def item(self, time, sfx="ding", filter_name="none", moment="surprise"):
        return {"time": time, "duration": .6, "moment_type": moment, "sfx": sfx, "filter": filter_name, "overlay": "", "enabled": True}

    def test_effect_spacing_and_repeated_sfx_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_effects_plan([self.item(0), self.item(3)], 20)
        with self.assertRaises(ValueError):
            validate_effects_plan([self.item(0), self.item(5)], 20)

    def test_enum_validation_rejects_unknown_moment(self):
        with self.assertRaises(Exception):
            EffectItem.model_validate(self.item(0, moment="fireworks"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for audio-only source test")
    def test_audio_only_download_is_not_a_video_source(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "audio.webm"
            subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "libopus", str(audio)], check=True)
            self.assertIsNone(probe_video_dimensions(audio))

    def test_filter_string_has_timed_enable_window(self):
        graph = build_video_filter(self.item(2, filter_name="saturation"), "in", "out")
        self.assertIn("eq=saturation=1.12", graph)
        self.assertIn("enable=", graph)
        self.assertIn("between(t", graph)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required for render smoke test")
    def test_render_audio_and_video_durations_match(self):
        from app.services.clip_generator import generate_short
        from app.config import settings

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            output = root / "clip.mp4"
            ffmpeg = shutil.which("ffmpeg")
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=2", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2", "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source)], check=True)
            generate_short(source, [], 0, 2, output, root, effects=[])
            old_music = settings.background_music_path
            settings.background_music_path = str(source)
            try:
                generate_short(source, [], 0, 2, output, root, effects=[{"time": .5, "duration": .5, "moment_type": "surprise", "sfx": "ding", "filter": "saturation", "overlay": "", "enabled": True}])
            finally:
                settings.background_music_path = old_music
            probe = subprocess.run([shutil.which("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(output)], capture_output=True, text=True, check=True)
            import json
            streams = json.loads(probe.stdout)["streams"]
            durations = {item["codec_type"]: float(item["duration"]) for item in streams if item.get("codec_type") in {"audio", "video"}}
            self.assertIn("video", durations)
            self.assertIn("audio", durations)
            self.assertLessEqual(abs(durations["video"] - durations["audio"]), .15)

if __name__ == "__main__":
    unittest.main()
