from kvoicewalk import KVoiceWalk
import argparse
import warnings
import soundfile as sf
from speech_generator import SpeechGenerator
import os
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description="A random walk Kokoro voice cloner.")

    # Common required arguments
    parser.add_argument("--target_text", type=str, help="The words contained in the target audio file. Should be around 100-200 tokens (two sentences).")

    # Optional arguments
    parser.add_argument("--other_text", type=str,
                      help="A segment of text used to compare self similarity. Should be around 100-200 tokens.",
                      default="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale.")
    parser.add_argument("--voice_folder", type=str,
                      help="Path to the voices you want to use as part of the random walk.",
                      default="./voices")
    parser.add_argument("--interpolate_start",
                      help="Goes through an interpolation search step before random walking",
                      action='store_true')
    parser.add_argument("--population_limit", type=int,
                      help="Limits the amount of voices used as part of the random walk",
                      default=10)
    parser.add_argument("--step_limit", type=int,
                      help="Limits the amount of steps in the random walk",
                      default=10000)
    parser.add_argument("--output", type=str,
                      help="Filename for the generated output audio",
                      default="out.wav")
    parser.add_argument("--mode", choices=["walk", "hybrid", "anneal", "bayes"], default="walk",
                        help="Choose the search strategy: walk (default), or hybrid (PCA + CMA-ES).")

    # Arguments for random walk mode
    group_walk = parser.add_argument_group('Random Walk Mode')
    group_walk.add_argument("--target_audio", type=str,
                          help="Path to the target audio file. Must be 24000 Hz mono wav file.")
    group_walk.add_argument("--starting_voice", type=str,
                          help="Path to the starting voice tensor")

    # Arguments for test mode
    group_test = parser.add_argument_group('Test Mode')
    group_test.add_argument("--test_voice", type=str,
                          help="Path to the voice tensor you want to test")

    # Arguments for util mode
    group_util = parser.add_argument_group('Utility Mode')
    group_util.add_argument("--export_bin",
                      help='Exports target voices in the --voice_folder directory',
                      action='store_true')
    args = parser.parse_args()

    # Export Utility
    if args.export_bin:
        if not args.voice_folder:
            parser.error("--voice_folder is required to export a voices bin file")

        voices = {}
        for filename in os.listdir(args.voice_folder):
            if filename.endswith('.pt'):
                file_path = os.path.join(args.voice_folder, filename)
                voice = torch.load(file_path)
                voices[filename] = voice

        with open("voices.bin", "wb") as f:
            np.savez(f,**voices)

        return

    # Validate arguments based on mode
    if args.test_voice:
        # Test mode
        if not args.target_text:
            parser.error("--target_text is required when using --test_voice")

        speech_generator = SpeechGenerator()
        audio = speech_generator.generate_audio(args.target_text, args.test_voice)
        sf.write(args.output, audio, 24000)
    else:
        # Random walk mode
        if not args.target_audio:
            parser.error("--target_audio is required for random walk mode")
        if not args.target_text:
            parser.error("--target_text is required for random walk mode")

        ktb = KVoiceWalk(args.target_audio,
                        args.target_text,
                        args.other_text,
                        args.voice_folder,
                        args.interpolate_start,
                        args.population_limit,
                        args.starting_voice,
                        args.mode)
        ktb.run(args.step_limit)
if __name__ == "__main__":
    main()
