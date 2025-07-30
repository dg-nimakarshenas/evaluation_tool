import pandas as pd
import json
from google import genai
from openai import OpenAI
from typing import List, Dict, Any
from typing_extensions import Annotated
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import os
import time
import enum


class FeedbackAnalysis(BaseModel):
    theme: str = Field(
        ...,
        description="An identified theme from the feedback, such as 'Communication', 'Timeliness', etc."
    )
    sentiment: Annotated[
        int,
        Field(
            ...,
            ge=1,
            le=5,
            description="An integer sentiment score from 1 (very negative) to 5 (very positive) that correponds to the sentiment towards that theme in the feedback."
        )
    ]

class Results(BaseModel):
    results: List[FeedbackAnalysis]

class FinalThemes(BaseModel):
    theme: str = Field(
        ...,
        description="The theme that best captures the list of themes provided."
    )


class FeedbackCategoriser:
    def __init__(self, api_key, model_name="gpt-4.1-mini", temperature=0, resident_feedback=None, council_feedback=None, contractor_feedback=None, feedback_col="feedback"):
        self.llm = genai.Client(api_key=api_key) if "gemini" in api_key else OpenAI(api_key=api_key)
        self.llm_type = "gemini" if "gemini" in api_key else "openai"
        self.temperature = temperature
        self.feedback_col = feedback_col
        self.model_name = model_name
        self.resident_feedback = resident_feedback
        self.contractor_feedback = contractor_feedback
        self.council_feedback = council_feedback

    def _categorise_resident_feedback(self, feedback, existing_themes, append_system_prompt=""):
        """
        Categorizes resident feedback using Gemini, reusing existing themes where possible.
        """
        # Dynamically create guidance for the LLM based on whether themes already exist.
        if existing_themes:
            theme_guidance = f"""
            We have identified the following themes so far: {existing_themes}.
            Please categorise the new feedback. If you think an existing theme fits the feedback exactly, use that theme.
            If the feedback introduces a new concept not covered by the existing themes, you are free to create a new, descriptive theme.
            Do not force a fit if it is not appropriate.
            """
        else:
            theme_guidance = "You are processing the first piece of feedback. You will establish the initial themes."
        
        # Construct the prompt, now including the dynamic theme guidance
        prompt = f"""You are an expert economic analyst, you re analysing the feedback from residents within social housing whose homes had been recently selected for a retrofit scheme. The feedback is wide ranging and covers all aspects
        of the retrofit process. You now want to get a comprehensive evaluation of the retrofit scheme from the perspective of the residents. You are to categorise the feedback into a set of themes and assign a sentiment score to each theme. 
        The sentiment score is a number between 1 and 5, where 1 is very negative and 5 is very positive. You are to return the results in JSON format. The JSON should be an array of objects, where each object has the following structure:
        \n```json\n{Results.model_json_schema()} for the following feedback:\n
                {feedback}\n\n
        {theme_guidance}        
                .\n\n {append_system_prompt} 

        Please note that the list of themes you provide should be comprehensive and covers all aspects of the feedback.
        \n\n
            """
        try:
            output_json = {} # Initialize an empty dict
            if self.llm_type == "gemini":
                response = self.llm.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                        'response_schema': Results.model_json_schema(),
                    },
                )
                categorized_feedback = response.text
                output_json = json.loads(categorized_feedback)
            
            elif self.llm_type == "openai":
                response = self.llm.responses.parse(
                    model=self.model_name,
                    input=prompt,
                    text_format=Results
                )
                # Assuming response.output_parsed gives the dict directly
                output_json = response.output_parsed 

            # Validate the data using the Pydantic model regardless of the LLM used
            validated_data = Results.model_validate(output_json)
            final_output = validated_data.model_dump(mode="json")

            return final_output.get("results", []) # Return the list of results or an empty list

        except Exception as e:  # Handle potential errors
            print(f"An error occurred: {e}") # It's good practice to log the error
            return f"Error: {e}"

    def categorise_all_resident_feedback(self, save: bool = False):
        """Iterates through all feedback, categorises it, and keeps track of themes."""
        self.resident_feedback["categorised_feedback"] = None  # Initialize the new column
        
        # Initialize a set to store unique themes across all feedback entries.
        self.existing_themes = set()

        for i, feedback in enumerate(self.resident_feedback[self.feedback_col]):
            print(f"Categorising feedback {i+1}/{len(self.resident_feedback)}")
            
            # Pass the current set of themes to the categorisation function.
            # We convert the set to a list for the prompt.
            categorised_results = self._categorise_resident_feedback(
                feedback=feedback,
                existing_themes=list(self.existing_themes)
            )
            
            self.resident_feedback.at[i, "categorised_feedback"] = categorised_results
            
            # Update the set of existing themes with any new ones from the current response.
            # This check ensures we only try to iterate over a valid list of results.
            if isinstance(categorised_results, list):
                for result in categorised_results:
                    if 'theme' in result:
                        self.existing_themes.add(result['theme'])
        if save:
            self.resident_feedback.to_excel("categorised_resident_feedback.xlsx", index=False)
            print("Categorised feedback saved to 'categorised_resident_feedback.excel'")

        
    def get_theme_embeddings(
        self,
        feedback_df: pd.DataFrame,
        model_name: str,
        batch_size: int = 20,
        save: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Generate embeddings for all themes in feedback_df using either Google Gemini or OpenAI,
        batching requests to avoid exceeding API limits.

        Args:
            feedback_df: DataFrame containing your feedback data.
            model_name: The embedding model to use (e.g., "gemini-embedding-exp-03-07" for Gemini,
                        "text-embedding-ada-002" for OpenAI).
            batch_size: Number of themes to send per request.
            save: Whether to save the embeddings to a JSON file.

        Returns:
            A dict mapping each theme (str) to its embedding vector (List[float]).
        """
        themes = self._get_all_themes(feedback_df=feedback_df)
        all_embeddings: Dict[str, List[float]] = {}

        size = len(themes)
        num_batches = int(size / batch_size) + (1 if size % batch_size else 0)

        # Process the data in chunks
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch = themes[start_idx:end_idx]
            batch_size_processed = len(batch)

            if batch_size_processed <= 0:
                break

            if self.llm_type == "gemini":
                response = self.llm.models.embed_content(
                    model=model_name,
                    contents=batch
                )
                # Extract the raw float vectors for Gemini
                for theme, emb_obj in zip(batch, response.embeddings):
                    if hasattr(emb_obj, "values"):
                        vec = emb_obj.values
                    else:
                        # Fallback for other possible structures
                        vec = emb_obj
                    all_embeddings[theme] = vec

            elif self.llm_type == "openai":
                response = self.llm.embeddings.create(
                    model=model_name,
                    input=batch
                )
                # Extract the raw float vectors for OpenAI
                for j, theme in enumerate(batch):
                    all_embeddings[theme] = response.data[j].embedding
            else:
                raise ValueError(f"Unsupported llm_type: {self.llm_type}")


        if save:
            with open("theme_embeddings.json", "w") as f:
                json.dump(all_embeddings, f, indent=4)
        return all_embeddings

    def _get_all_themes(self, feedback_df: pd.DataFrame) -> List[str]:
        """Extracts all unique themes from the resident feedback."""
        all_themes = set()
        # Ensure the column exists before iterating
        if "categorised_feedback" in feedback_df.columns:
            for feedback in feedback_df["categorised_feedback"]:
                if feedback:
                    # Use a safer method than eval
                    try:
                        # If feedback is a string representation of a list of dicts
                        if isinstance(feedback, str):
                            feedback = json.loads(feedback.replace("'", '"'))
                        
                        # Process if it is a list
                        if isinstance(feedback, list):
                            for item in feedback:
                                if isinstance(item, dict) and "theme" in item:
                                    all_themes.add(item["theme"])
                    except (json.JSONDecodeError, TypeError):
                        # Handle cases where feedback is not in the expected format
                        # You might want to log these instances
                        pass
        return list(all_themes)

    def plot_knn_loss_curve(
        self,
        embeddings: Dict[str, List[float]],
        k_min: int = 1,
        k_max: int = 20,
        n_clusters: int = 5,
        random_state: int = 42,
        annotate: bool = True,
        annotate_fontsize: int = 8
        ):
        """
        1. Plot the elbow curve (inertia vs. number of clusters K from k_min to k_max).
        2. Reduce embeddings to 2D via PCA and plot KMeans clusters with colour and annotations.
        
        Args:
            embeddings: Mapping from item (e.g., theme) to its embedding vector.
            k_min: Minimum K for the elbow curve.
            k_max: Maximum K for the elbow curve.
            n_clusters: Number of clusters for the 2D scatter.
            random_state: Random seed for reproducibility.
            annotate: Whether to annotate points with their labels.
            annotate_fontsize: Font size for the annotations.
        """
        # Prepare data
        themes = list(embeddings.keys())
        X = list(embeddings.values())
        pca = PCA(n_components=2, random_state=random_state)
        X = pca.fit_transform(X)
        # 1. Elbow plot
        inertias = []
        k_values = list(range(k_min, k_max + 1))
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=random_state)
            km.fit(X)
            inertias.append(km.inertia_)
        
        plt.figure()
        plt.plot(k_values, inertias, marker='o')
        plt.title("Elbow Method for Optimal k")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia (Loss)")
        plt.xticks(k_values)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        plt.figure(figsize=(10, 7))
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.title(f"KMeans Clusters (k={n_clusters}) on 2D PCA Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if annotate:
            for (x, y), theme in zip(X, themes):
                plt.annotate(theme, (x, y), fontsize=annotate_fontsize, alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    
    def plot_hierarchical_dendrogram(
            self,
            embeddings: Dict[str, List[float]],
            method: str = "ward",
            truncate_mode: str = None,
            p: int = 12,
            max_clusters: int = None,  # Optional number of clusters to extract
            distance_threshold: float = None  # Optional distance threshold to cut the tree
        ):
        """
        Perform agglomerative clustering on the provided embeddings and plot a dendrogram to help decide where to cut.
        Optionally extract clusters at a given level (either by distance or number of clusters).
        
        Args:
            embeddings: Dict mapping each item (e.g., theme) to its embedding vector.
            method: Linkage method ('ward', 'average', 'complete', etc.).
            truncate_mode: 'lastp', 'level', or None for a full tree.
            p: Number of last clusters to show if truncate_mode='lastp'.
            max_clusters: The number of clusters to extract (optional, overrides distance_threshold).
            distance_threshold: The distance threshold to cut the tree (optional, overrides max_clusters).
            
        Returns:
            cluster_labels: List of cluster assignments corresponding to each theme.
        """
        labels = list(embeddings.keys())
        X = list(embeddings.values())

        # Perform hierarchical clustering
        Z = linkage(X, method=method)

        # Plot the dendrogram
        plt.figure(figsize=(10, 6))
        dendrogram(
            Z,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=10,
            truncate_mode=truncate_mode,
            p=p
        )
        plt.title(f"Hierarchical Clustering Dendrogram ({method})")
        plt.xlabel("Themes")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.show()

        # Extract clusters based on the specified level (either number of clusters or distance threshold)
        if max_clusters:
            cluster_labels = fcluster(Z, t=max_clusters, criterion='maxclust')  # Cut at a specific number of clusters
        elif distance_threshold:
            cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')  # Cut at a specific distance
        else:
            cluster_labels = fcluster(Z, t=2, criterion='maxclust')  # Default: split into 2 clusters
        
        # Return the cluster labels for each theme
        return cluster_labels
    
    def assign_final_themes_to_clusters(
        self,
        embeddings: Dict[str, List[float]],
        cluster_indices: List[int],
        model_name: str = None,
    ) -> Dict[int, str]:
        """
        Given a dict of theme embeddings and a parallel list of cluster assignments,
        group the themes by cluster, then ask the LLM to give each cluster a single
        overarching theme label.

        Args:
            embeddings: Dict mapping each theme to its embedding vector.
            cluster_indices: List of cluster IDs, one per theme, in the same order
                             as embeddings.keys().
            model_name:     Which model to use for summarization (optional, defaults to self.model_name).

        Returns:
            Dict mapping each cluster ID to its final label (str), and theme_to_cluster mapping.
        """
        themes = list(embeddings.keys())
        if len(themes) != len(cluster_indices):
            raise ValueError("Number of themes and cluster indices must match.")
        
        theme_to_cluster: Dict[str, int] = {
            theme: cid for theme, cid in zip(themes, cluster_indices)
        }

        clusters: Dict[int, List[str]] = {}
        for theme, cid in zip(themes, cluster_indices):
            clusters.setdefault(cid, []).append(theme)

        model_name = model_name or self.model_name
        final_labels: Dict[int, str] = {}

        for cid, theme_list in clusters.items():
            prompt = f"""
You are an expert summarizer of topic lists. I’ve grouped a set of related themes below.
Please suggest *one* concise, descriptive theme label that best captures all of them.

Cluster ID: {cid}
Themes:
{json.dumps(theme_list, indent=2)}

Respond with just the label—no JSON, no extra commentary.
"""
            if self.llm_type == "gemini":
                resp = self.llm.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={"response_mime_type": "text/plain"},
                )
                label = resp.text.strip()
            elif self.llm_type == "openai":
                response = self.llm.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert summarizer of topic lists."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=32,
                )
                label = response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported llm_type: {self.llm_type}")

            final_labels[cid] = label

        return final_labels, theme_to_cluster

    def apply_final_themes(
            self,
            df: pd.DataFrame,
            final_labels: Dict[int, str],
            theme_to_cluster: Dict[str, int]
        ) -> pd.DataFrame:
            """
            Adds/appends a new column "final_feedback" to df where each row is a list of dicts:
            {
                "final_theme": <cluster_label>,
                "sentiment": <original sentiment>
            }

            Args:
            df: your DataFrame
            final_labels: mapping {cluster_id: final_theme_label}
            theme_to_cluster: mapping {original_theme: cluster_id}
            """

            def remap(themes_list: List[Dict[str, Any]]):
                remapped = []
                themes_list = eval(themes_list) if isinstance(themes_list, str) else themes_list
                for item in themes_list:
                    orig_theme = item["theme"]
                    sentiment = item["sentiment"]

                    cid = theme_to_cluster.get(orig_theme)
                    if cid is None:
                        # fallback if a theme somehow wasn’t clustered
                        final = orig_theme
                    else:
                        final = final_labels[cid]

                    remapped.append({
                        "final_theme": final,
                        "sentiment": sentiment
                    })
                return remapped

            df = df.copy()
            df["final_feedback"] = df["categorised_feedback"].apply(remap)
            return df
    
    def save_resident_feedback(self, filename: str):
        """Saves the resident feedback DataFrame to a CSV file."""
        self.resident_feedback.to_csv(filename, index=False) if filename.endswith('.csv') else self.resident_feedback.to_excel(filename, index=False)
        print(f"Feedback saved to {filename}")    

if __name__ == "__main__":
    resident_feedback = pd.read_excel("data\\property_summaries_with_synthesised_feedback.xlsx")
    feedback_categoriser = FeedbackCategoriser(api_key=os.environ.get("OPENAI_API_KEY"), model_name="gpt-4.1-mini", resident_feedback=resident_feedback, feedback_col="Resident_Feedback")
    feedback_categoriser.categorise_all_resident_feedback(save=True)  # Set save=True to save the categorised feedback
    feedback_categoriser.llm_type = "openai"  # Set to "gemini" or "openai" based on your API key
    embeddings = feedback_categoriser.get_theme_embeddings(feedback_df=feedback_categoriser.resident_feedback, model_name="text-embedding-3-small")
    clusters = feedback_categoriser.plot_hierarchical_dendrogram(embeddings, method="ward", max_clusters=10)
    final_labels = feedback_categoriser.assign_final_themes_to_clusters(embeddings, clusters)
    feedback_categoriser.resident_feedback = feedback_categoriser.apply_final_themes(feedback_categoriser.resident_feedback, final_labels[0], final_labels[1])
    feedback_categoriser.save_resident_feedback("categorised_resident_feedback.xlsx")
    print(clusters)
