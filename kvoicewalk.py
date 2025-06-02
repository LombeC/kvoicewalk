
from typing import Any
from fitness_scorer import FitnessScorer
from initial_selector import InitialSelector
from speech_generator import SpeechGenerator
from voice_generator import VoiceGenerator
import random
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torch
import os

OUT_DIR = os.environ.get("KVOICEWALK_OUT_DIR","./out")

class KVoiceWalk:
    def __init__(self,target_audio: str,target_text: str,other_text:str,voice_folder:str,interpolate_start: bool,
                 population_limit: int, starting_voice: str, mode: str) -> None:
        self.target_text = target_text
        self.other_text = other_text
        self.initial_selector = InitialSelector(target_audio,target_text,other_text,voice_folder=voice_folder)
        voices: list[torch.Tensor] = []
        if interpolate_start:
            voices = self.initial_selector.interpolate_search(population_limit)
        else:
            voices = self.initial_selector.top_performer_start(population_limit)
        self.speech_generator = SpeechGenerator()
        self.fitness_scorer = FitnessScorer(target_audio)
        self.voice_generator = VoiceGenerator(voices,starting_voice)
        # Either the mean or the supplied voice tensor
        self.starting_voice = self.voice_generator.starting_voice
        self.mode = mode

    def run(self, step_limit: int):
        if self.mode == "walk":
            self.random_walk(step_limit)
        elif self.mode == "hybrid":
            self.hybrid_search(step_limit)
        elif self.mode == "anneal":
            self.random_walk_with_simulated_annealing(step_limit)
        elif self.mode == "bayes":
            self.bayesian_opt_search(step_limit)
        elif self.mode == "mixed":
            self.mixed_strategy_search(step_limit)
        elif self.mode == "genetic":
            self.genetic_search(step_limit)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def hybrid_search(self, step_limit: int):
        import numpy as np
        from sklearn.decomposition import PCA

        tqdm.write(">> Starting PCA-guided initialization...")

        os.makedirs(OUT_DIR, exist_ok=True)

        # Score the actual starting voice
        best_voice = self.starting_voice
        original_shape = best_voice.shape
        best_results = self.score_voice(best_voice)
        best_vec = best_voice.view(-1).numpy()

        tqdm.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}')

        # Prepare PCA on top performer voices
        stacked = torch.stack(self.voice_generator.voices).view(len(self.voice_generator.voices), -1).numpy()
        pca = PCA(n_components=4)
        pca.fit(stacked)

        total_iters = pca.n_components_ * (step_limit // pca.n_components_)
        pbar = tqdm(total=total_iters, desc="Hybrid Search")

        step = 0
        for dim in range(pca.n_components_):
            for alpha in np.linspace(-2, 2, num=step_limit // pca.n_components_):
                vec = best_vec + alpha * pca.components_[dim]
                voice = torch.tensor(vec, dtype=torch.float32).view(original_shape)

                min_similarity = best_results["target_similarity"] * 0.98
                results = self.score_voice(voice, min_similarity)

                if results["score"] > best_results["score"]:
                    best_results = results
                    best_voice = voice
                    tqdm.write(f'Step:{step:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} PCA Dim:{dim} α:{alpha:.2f}')

                    torch.save(best_voice, f'{OUT_DIR}/{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{step}.pt')
                    sf.write(f'{OUT_DIR}/{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{step}.wav', best_results["audio"], 24000)

                step += 1
                pbar.update(1)

        pbar.close()
        tqdm.write(f">> Hybrid search complete. Best Score: {best_results['score']:.2f}")

    def random_walk(self,step_limit: int):
        os.makedirs(OUT_DIR,exist_ok=True)

        # Score Initial Voice
        best_voice = self.starting_voice
        best_results = self.score_voice(self.starting_voice)
        tqdm.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}')

        # Random Walk Loop
        for i in tqdm(range(step_limit)):
            # TODO: Expose to CLI
            diversity = random.uniform(0.01,0.15)
            voice = self.voice_generator.generate_voice(best_voice,diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            voice_results = self.score_voice(voice,min_similarity)

            # Set new winner if score is better
            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                best_voice = voice
                tqdm.write(f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}')
                # Save results so folks can listen
                torch.save(best_voice, f'{OUT_DIR}/{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{i}.pt')
                sf.write(f'{OUT_DIR}/{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{i}.wav', best_results["audio"], 24000)

    def random_walk_with_simulated_annealing(self, step_limit: int):
        import math

        os.makedirs(OUT_DIR, exist_ok=True)

        best_voice = self.starting_voice
        best_results = self.score_voice(best_voice)
        best_score = best_results["score"]
        tqdm.write(f'>> Starting Random Walk with Simulated Annealing...')
        tqdm.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_score:.2f}')

        T_init = 1.0      # Initial temperature
        T_final = 0.01    # Final temperature
        alpha = 0.95      # Cooling rate

        T = T_init

        for i in tqdm(range(step_limit), desc="Simulated Annealing"):
            diversity = random.uniform(0.01, 0.15)
            new_voice = self.voice_generator.generate_voice(best_voice, diversity)

            min_similarity = best_results["target_similarity"] * 0.98
            new_results = self.score_voice(new_voice, min_similarity)
            new_score = new_results["score"]

            accept = False
            if new_score > best_score:
                accept = True
            else:
                delta = new_score - best_score
                probability = math.exp(delta / T)
                if random.random() < probability:
                    accept = True

            if accept:
                best_voice = new_voice
                best_results = new_results
                best_score = new_score
                tqdm.write(f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_score:.2f} Diversity:{diversity:.2f} T:{T:.4f}')

                torch.save(best_voice, f'{OUT_DIR}/{best_score:.2f}_{best_results["target_similarity"]:.2f}_{i}.pt')
                sf.write(f'{OUT_DIR}/{best_score:.2f}_{best_results["target_similarity"]:.2f}_{i}.wav', best_results["audio"], 24000)

            # Cool the temperature
            T = max(T_final, T * alpha)

    def bayesian_opt_search(self, step_limit: int):
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args

        tqdm.write(">> Starting Bayesian Optimization in raw latent space...")

        original_shape = self.starting_voice.shape
        latent_dim = self.starting_voice.numel()
        starting_vec = self.starting_voice.view(-1).numpy()

        # Search space for each dimension (can tighten bounds later)
        space = [Real(-2.0, 2.0, name=f"dim_{i}") for i in range(latent_dim)]

        best_score = -1
        best_results = None
        best_voice = self.starting_voice

        @use_named_args(space)
        def objective(**params):
            vec = np.array([params[f"dim_{i}"] for i in range(latent_dim)], dtype=np.float32)
            voice = torch.tensor(vec).view(original_shape)
            results = self.score_voice(voice)
            nonlocal best_score, best_results, best_voice

            if results["score"] > best_score:
                best_score = results["score"]
                best_results = results
                best_voice = voice
                tqdm.write(f'New Best → Score: {best_score:.2f} | Target Sim: {results["target_similarity"]:.3f} | Self Sim: {results["self_similarity"]:.3f} | Feature Sim: {results["feature_similarity"]:.3f}')
                sf.write(f'{OUT_DIR}/bo_best_{best_score:.2f}.wav', results["audio"], 24000)
                torch.save(voice, f'{OUT_DIR}/bo_best_{best_score:.2f}.pt')

            return -results["score"]  # because gp_minimize does minimization

        # Run the actual optimization
        gp_minimize(
            objective,
            space,
            n_calls=step_limit,
            x0=[starting_vec],
            random_state=42
        )

        tqdm.write(f">> Bayesian Optimization complete. Final Score: {best_score:.2f}")

    def mixed_strategy_search(self, step_limit: int):
        from sklearn.decomposition import PCA
        from skopt import gp_minimize
        from skopt.space import Real
        from skopt.utils import use_named_args
        import numpy as np

        tqdm.write(">> Phase 1: PCA-guided coarse exploration...")

        stacked = torch.stack(self.voice_generator.voices).view(len(self.voice_generator.voices), -1).numpy()
        pca = PCA(n_components=4)
        pca.fit(stacked)

        best_score = -1
        best_voice = None
        best_results = None
        original_shape = self.voice_generator.voices[0].shape

        for dim in range(pca.n_components_):
            for alpha in np.linspace(-2, 2, num=8):
                vec = pca.mean_ + alpha * pca.components_[dim]
                voice_tensor = torch.tensor(vec, dtype=torch.float32).view(original_shape)
                results = self.score_voice(voice_tensor)
                score = results["score"]
                if score > best_score:
                    best_score = score
                    best_voice = voice_tensor
                    best_results = results
                    tqdm.write(f'PCA → Score: {score:.2f} | Target: {results["target_similarity"]:.3f}')
                    sf.write(f'{OUT_DIR}/pca_best_{score:.2f}.wav', results["audio"], 24000)
                    torch.save(voice_tensor, f'{OUT_DIR}/pca_best_{score:.2f}.pt')

        tqdm.write(">> Phase 2: Bayesian Optimization...")

        latent_dim = best_voice.numel()
        best_vec = best_voice.view(-1).numpy()
        space = [Real(-2.0, 2.0, name=f"dim_{i}") for i in range(latent_dim)]

        @use_named_args(space)
        def objective(**params):
            vec = np.array([params[f"dim_{i}"] for i in range(latent_dim)], dtype=np.float32)
            voice = torch.tensor(vec).view(original_shape)
            results = self.score_voice(voice)
            nonlocal best_score, best_voice, best_results
            if results["score"] > best_score:
                best_score = results["score"]
                best_voice = voice
                best_results = results
                tqdm.write(f'Bayes → Score: {best_score:.2f} | Target: {results["target_similarity"]:.3f}')
                sf.write(f'{OUT_DIR}/bayes_best_{best_score:.2f}.wav', results["audio"], 24000)
                torch.save(voice, f'{OUT_DIR}/bayes_best_{best_score:.2f}.pt')
            return -results["score"]

        gp_minimize(objective, space, n_calls=step_limit // 2, x0=[best_vec], random_state=42)

        tqdm.write(">> Phase 3: Simulated Annealing...")

        voice = best_voice
        results = best_results

        for i in tqdm(range(step_limit // 2)):
            temp = max(0.01, 1.0 - (i / (step_limit // 2)))
            diversity = temp * 0.25  # scale range over time
            new_voice = self.voice_generator.generate_voice(voice, diversity)
            new_results = self.score_voice(new_voice, results["target_similarity"] * 0.98)

            delta = new_results["score"] - results["score"]
            accept_prob = np.exp(delta / temp) if delta < 0 else 1.0

            if np.random.rand() < accept_prob:
                voice = new_voice
                results = new_results
                if results["score"] > best_score:
                    best_score = results["score"]
                    best_voice = voice
                    best_results = results
                    tqdm.write(f'SA Step {i} → Score: {best_score:.2f} | Target: {results["target_similarity"]:.3f}')
                    sf.write(f'{OUT_DIR}/sa_best_{best_score:.2f}.wav', results["audio"], 24000)
                    torch.save(voice, f'{OUT_DIR}/sa_best_{best_score:.2f}.pt')

        tqdm.write(f">> Mixed strategy complete. Final Score: {best_score:.2f}")

    def genetic_search(self, step_limit: int):
        import numpy as np
        tqdm.write(">> Starting Genetic Algorithm search...")

        population_size = 16
        retain_top_k = 6
        mutation_rate = 0.1
        generations = step_limit

        original_shape = self.voice_generator.voices[0].shape
        dim = self.voice_generator.voices[0].numel()

        # Initialize population from starting voice with noise
        base_voice = self.starting_voice.view(-1)
        population = [
            base_voice + torch.randn_like(base_voice) * 0.05
            for _ in range(population_size)
        ]

        def crossover(parent1, parent2):
            mask = torch.rand_like(parent1) > 0.5
            return torch.where(mask, parent1, parent2)

        def mutate(tensor, rate):
            noise = torch.randn_like(tensor) * rate
            return tensor + noise

        best_score = -1
        best_voice = None
        best_results = None

        for gen in tqdm(range(generations)):
            scored = []
            for i, latent in enumerate(population):
                voice_tensor = latent.view(original_shape)
                results = self.score_voice(voice_tensor)
                score = results["score"]
                scored.append((score, latent, results))
                if score > best_score:
                    best_score = score
                    best_voice = voice_tensor
                    best_results = results
                    tqdm.write(f'Gen {gen:<3} → Score: {score:.2f} | Target: {results["target_similarity"]:.3f}')
                    sf.write(f'{OUT_DIR}/ga_best_{score:.2f}.wav', results["audio"], 24000)
                    torch.save(voice_tensor, f'{OUT_DIR}/ga_best_{score:.2f}.pt')

            # Select top performers
            scored.sort(reverse=True, key=lambda x: x[0])
            parents = [latent for _, latent, _ in scored[:retain_top_k]]

            # Reproduce
            next_population = []
            while len(next_population) < population_size:
                p1, p2 = random.sample(parents, 2)
                child = crossover(p1, p2)
                child = mutate(child, mutation_rate)
                next_population.append(child)

            population = next_population

        tqdm.write(f">> Genetic algorithm complete. Final score: {best_score:.2f}")

    def score_voice(self,voice: torch.Tensor,min_similarity: float = 0.0) -> dict[str,Any]:
        """Using a harmonic mean calculation to provide a score for the voice in similarity"""
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        results: dict[str,Any] = {
            'audio': audio
        }
        # Bail early and save the compute if the similarity sucks
        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(self.fitness_scorer.hybrid_similarity(audio,audio2,target_similarity))
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity

        return results
