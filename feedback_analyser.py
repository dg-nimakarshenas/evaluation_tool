import pandas as pd
import json
from google import genai
from typing import List, Dict, Any
from typing_extensions import Annotated
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import time
import enum


class FeedbackAnalysis(BaseModel):
    theme: str = Field(
        ...,
        description="An identified theme from the feedback, such as 'communication', 'timeliness', etc."
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
    def __init__(self, api_key, model_name="gemini-2.5-flash-preview-04-17", temperature=0, resident_feedback=None, council_feedback=None, contractor_feedback=None):
        self.llm = genai.Client(api_key=api_key)
        self.temperature = temperature
        self.model_name = model_name
        self.resident_feedback = resident_feedback
        self.contractor_feedback = contractor_feedback
        self.council_feedback = council_feedback

    def _categorise_resident_feedback(self, feedback, append_system_prompt=""):
        """Categorizes resident feedback using Gemini and enforces structure."""
        # Construct the input list with each repair log numbered

        prompt = f"""You are an expert economic analyst, you re analysing the feedback from residents within social housing whose homes had been recently selected for a retrofit scheme. The feedback is wide ranging and covers all aspects
        of the retrofit process. You now want to get a comprehensive evaluation of the retrofit scheme from the perspective of the residents. You are to categorise the feedback into a set of themes and assign a sentiment score to each theme. 
        The sentiment score is a number between 1 and 5, where 1 is very negative and 5 is very positive. You are to return the results in JSON format. The JSON should be an array of objects, where each object has the following structure:
        \n```json\n{Results.model_json_schema()} for the following feedback:\n
                {feedback}\n\n
            .\n\n {append_system_prompt} 

        Please note that you are not limited to any set of themes, we have left this open for you to decide to prevent bias. Please ensure that the list of themes you provide is comprehensive and covers all aspects of the feedback.
        \n\n
            """
        try:
            response = self.llm.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': Results.model_json_schema(),
                },
            )

            # Parse the JSON string into Pydantic objects
            categorized_feedback = response.text  # access the list field in the container
            output_json = json.loads(categorized_feedback)

            return output_json["results"]

        except Exception as e:  # Handle potential errors (API issues, parsing problems, etc.)
            return f"Error: {e}"
        
    def categorise_all_resident_feedback(self):
        self.resident_feedback["categorised_feedback"] = None  # Initialize the new column
        for i, feedback in enumerate(self.resident_feedback["feedback"]):
            print(f"Categorising feedback {i+1}/{len(self.resident_feedback)}")
            self.resident_feedback.at[i, "categorised_feedback"] = self._categorise_resident_feedback(feedback=feedback)
        
    def get_theme_embeddings(
            self,
            feedback_df: pd.DataFrame,
            model_name: str = "gemini-embedding-exp-03-07",
            batch_size: int = 20,
            save: bool = False,
        ) -> Dict[str, List[float]]:
        """
        Generate embeddings for all themes in feedback_df using Google Gemini, batching
        requests to avoid exceeding the 250-input limit.

        Args:
            feedback_df: List of dicts containing your feedback data.
            model_name: The Gemini embedding model to use.
            batch_size: Number of themes to send per request (max 250 per API spec).

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
            batch_size = len(batch)
            if batch_size <= 0:
                break
            time.sleep(30)  # Rate limit to avoid hitting API limits
            response = self.llm.models.embed_content(
                model=model_name,
                contents=batch
            )
            # Extract the raw float vectors
            for theme, emb_obj in zip(batch, response.embeddings):
                # Try both common attribute names:
                if hasattr(emb_obj, "embedding"):
                    vec = emb_obj.embedding
                elif hasattr(emb_obj, "values"):
                    vec = emb_obj.values
                else:
                    # fallback: assume it's already a list
                    vec = emb_obj  # type: ignore

                all_embeddings[theme] = vec  # now a List[float]
        if save:
            with open("theme_embeddings.json", "w") as f:
                json.dump(all_embeddings, f, indent=4)
        return all_embeddings
    
    def _get_all_themes(self, feedback_df: pd.DataFrame) -> List[str]:
        """Extracts all unique themes from the resident feedback."""
        all_themes = set()
        for feedback in feedback_df["categorised_feedback"]:
            if feedback:
                feedback = eval(feedback) if isinstance(feedback, str) else feedback
                for item in feedback:
                    all_themes.add(item["theme"])
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
        model_name: str = "gemini-2.5-flash-preview-04-17",
    ) -> Dict[int, str]:
        """
        Given a dict of theme embeddings and a parallel list of cluster assignments,
        group the themes by cluster, then ask Gemini to give each cluster a single
        overarching theme label.

        Args:
            embeddings: Dict mapping each theme to its embedding vector.
            cluster_indices: List of cluster IDs, one per theme, in the same order
                             as embeddings.keys().
            model_name:     Which Gemini model to use for summarization.

        Returns:
            Dict mapping each cluster ID to its final label (str).
        """
        # 1) Align the flat list of theme names with cluster IDs
        themes = list(embeddings.keys())
        if len(themes) != len(cluster_indices):
            raise ValueError("Number of themes and cluster indices must match.")
        
        theme_to_cluster: Dict[str, int] = {
            theme: cid for theme, cid in zip(themes, cluster_indices)
        }

        # 2) Group theme names by cluster
        clusters: Dict[int, List[str]] = {}
        for theme, cid in zip(themes, cluster_indices):
            clusters.setdefault(cid, []).append(theme)

        # 3) For each cluster, prompt Gemini for a single covering theme
        llm = self.llm  # your genai.Client instance
        final_labels: Dict[int, str] = {}

        for cid, theme_list in clusters.items():
            prompt = f"""
You are an expert summarizer of topic lists.  I’ve grouped a set of related themes below.
Please suggest *one* concise, descriptive theme label that best captures all of them.

Cluster ID: {cid}
Themes:
{json.dumps(theme_list, indent=2)}

Respond with just the label—no JSON, no extra commentary.
"""
            resp = llm.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"response_mime_type": "text/plain"},
            )
            label = resp.text.strip()
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
        self.resident_feedback.to_csv(filename, index=False)
        print(f"Feedback saved to {filename}")    

if __name__ == "__main__":
    gemini_api_key="AIzaSyCAHoYzCRaQiUoncfeMn36n5SyhFlNJ_7s"
    resident_feedback = pd.read_csv("categorised_resident_feedback.csv")
    feedback_categoriser = FeedbackCategoriser(api_key=gemini_api_key, resident_feedback=resident_feedback)
    #feedback_categoriser.categorise_all_resident_feedback()
    embeddings = feedback_categoriser.get_theme_embeddings(feedback_df=feedback_categoriser.resident_feedback)
    clusters = feedback_categoriser.plot_hierarchical_dendrogram(embeddings, method="ward", max_clusters=10)
    final_labels = feedback_categoriser.assign_final_themes_to_clusters(embeddings, clusters)
    feedback_categoriser.resident_feedback = feedback_categoriser.apply_final_themes(feedback_categoriser.resident_feedback, final_labels[0], final_labels[1])
    feedback_categoriser.save_resident_feedback("categorised_resident_feedback.csv")
    print(clusters)
