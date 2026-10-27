from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


class GATBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        *,
        concat: bool = True,
        edge_dim: int | None = None,
        dropout: float = 0.4,
        residual: bool = True,
        norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_dim,
            out_dim,
            heads=heads,
            concat=concat,
            dropout=dropout,
            edge_dim=edge_dim,
        )
        self.dropout = dropout
        self.residual = residual
        output_dim = out_dim * heads if concat else out_dim
        self.norm = nn.LayerNorm(output_dim) if norm else None
        self.proj = (
            nn.Linear(in_dim, output_dim)
            if residual and in_dim != output_dim
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.conv(x, edge_index, edge_attr)
        if self.residual:
            residual = self.proj(x) if self.proj is not None else x
            hidden = hidden + residual
        if self.norm is not None:
            hidden = self.norm(hidden)
        return F.elu(hidden)


class GATRanker(nn.Module):
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        *,
        hidden: int = 64,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.4,
        residual: bool = True,
        norm: bool = True,
        use_edge: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge = use_edge
        edge_dim = edge_in if use_edge else None
        self.blocks = nn.ModuleList(
            [
                GATBlock(
                    node_in,
                    hidden,
                    heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    residual=residual,
                    norm=norm,
                )
            ]
        )
        for _ in range(max(0, layers - 2)):
            self.blocks.append(
                GATBlock(
                    hidden * heads,
                    hidden,
                    heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    residual=residual,
                    norm=norm,
                )
            )
        self.blocks.append(
            GATBlock(
                hidden * heads,
                1,
                1,
                concat=False,
                edge_dim=edge_dim,
                dropout=dropout,
                residual=False,
                norm=False,
            )
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, edge_index, edge_attr if self.use_edge else None)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x.squeeze(-1)


class GraphChatEngine:
    def __init__(
        self,
        *,
        artifacts_dir: str | Path,
        checkpoint_path: str | Path,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.embedding_model_name = embedding_model
        self.llm_base_url = (
            llm_base_url
            or os.getenv("LLM_BASE_URL")
            or "http://host.docker.internal:1234/v1"
        ).rstrip("/")
        self.llm_model = (
            llm_model
            or os.getenv("LLM_MODEL")
            or "qwen2.5-coder-14b-instruct"
        )
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "local")
        self.device = torch.device(os.getenv("GAT_DEVICE", "cpu"))
        torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))

        self.embedder = SentenceTransformer(embedding_model, device="cpu")
        self._load_graph()
        self._load_ranker()

    def _load_graph(self) -> None:
        entities_path = self.artifacts_dir / "entities.parquet"
        relationships_path = self.artifacts_dir / "relationships.parquet"
        if not entities_path.exists() or not relationships_path.exists():
            raise FileNotFoundError(
                "Не найдены entities.parquet и relationships.parquet в "
                f"{self.artifacts_dir}"
            )

        self.entities = pd.read_parquet(entities_path).reset_index(drop=True)
        relationships = pd.read_parquet(relationships_path).reset_index(drop=True)
        self.title_col = "title" if "title" in self.entities else "name"
        self.text_col = next(
            column
            for column in ("description", "text", "summary")
            if column in self.entities
        )
        self.entities["_title_norm"] = (
            self.entities[self.title_col].astype(str).str.strip().str.lower()
        )

        node_texts = (
            self.entities[self.title_col].fillna("").astype(str)
            + ". "
            + self.entities[self.text_col].fillna("").astype(str)
        ).tolist()
        node_embeddings = self.embedder.encode(
            node_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        title_to_idx = {
            title: index
            for index, title in enumerate(self.entities["_title_norm"])
        }
        id_to_idx: dict[str, int] = {}
        for column in ("id", "human_readable_id"):
            if column in self.entities:
                for index, value in enumerate(self.entities[column].astype(str)):
                    id_to_idx[value] = index

        source_col = next(
            column for column in ("source", "src", "head") if column in relationships
        )
        target_col = next(
            column for column in ("target", "dst", "tail") if column in relationships
        )

        def resolve(value: Any) -> int:
            text = str(value).strip()
            return title_to_idx.get(text.lower(), id_to_idx.get(text, -1))

        source = relationships[source_col].map(resolve).to_numpy()
        target = relationships[target_col].map(resolve).to_numpy()
        valid = (source >= 0) & (target >= 0)
        source = source[valid].astype(int)
        target = target[valid].astype(int)
        self.relationships = relationships.loc[valid].copy().reset_index(drop=True)
        self.relationships["source_idx"] = source
        self.relationships["target_idx"] = target

        relation_descriptions = (
            self.relationships["description"].fillna("").astype(str).tolist()
            if "description" in self.relationships
            else [""] * len(self.relationships)
        )
        relation_types = (
            self.relationships["type"].fillna("related").astype(str).tolist()
            if "type" in self.relationships
            else ["related"] * len(self.relationships)
        )
        type_vocabulary = sorted(set(relation_types)) or ["related"]
        type_to_index = {value: index for index, value in enumerate(type_vocabulary)}
        type_one_hot = torch.zeros(len(relation_types), len(type_vocabulary))
        for index, relation_type in enumerate(relation_types):
            type_one_hot[index, type_to_index[relation_type]] = 1.0

        relation_embeddings = self.embedder.encode(
            relation_descriptions,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        relation_embeddings_tensor = torch.tensor(
            relation_embeddings, dtype=torch.float32
        )
        forward_edge_attr = torch.cat(
            [type_one_hot, relation_embeddings_tensor], dim=1
        )
        forward_edges = torch.tensor(
            np.stack([source, target]), dtype=torch.long
        )
        self.graph = Data(
            x=torch.tensor(node_embeddings, dtype=torch.float32),
            edge_index=torch.cat([forward_edges, forward_edges.flip(0)], dim=1),
            edge_attr=torch.cat([forward_edge_attr, forward_edge_attr], dim=0),
            num_nodes=len(self.entities),
        )
        self.relation_embeddings = relation_embeddings_tensor
        self.title_to_idx = title_to_idx
        self.hop_one: dict[int, set[int]] = defaultdict(set)
        for source_idx, target_idx in zip(source, target):
            self.hop_one[int(source_idx)].add(int(target_idx))
            self.hop_one[int(target_idx)].add(int(source_idx))

    def _load_ranker(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Не найден checkpoint модели: {self.checkpoint_path}"
            )
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        config = checkpoint["config"]
        self.ranker = GATRanker(
            self.graph.x.size(1) + 2,
            self.graph.edge_attr.size(1) + 1,
            hidden=int(config["hidden"]),
            layers=int(config["layers"]),
            heads=int(config["heads"]),
            dropout=float(config["dropout"]),
            residual=bool(config["residual"]),
            norm=bool(config["norm"]),
            use_edge=bool(config["use_edge"]),
        ).to(self.device)
        self.ranker.load_state_dict(checkpoint["state_dict"], strict=True)
        self.ranker.eval()
        self.model_config = config

    def _embed_question(self, question: str) -> np.ndarray:
        return self.embedder.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _find_seeds(
        self, question: str, question_embedding: np.ndarray, count: int = 3
    ) -> list[int]:
        similarities = self.graph.x.numpy() @ question_embedding
        degrees = torch.bincount(
            self.graph.edge_index[0], minlength=self.graph.num_nodes
        ).numpy()
        scores = similarities - 0.06 * np.log1p(degrees)
        question_lower = question.lower()
        exact_boost = np.array(
            [
                1.0 if len(title) >= 3 and title in question_lower else 0.0
                for title in self.entities["_title_norm"].tolist()
            ]
        )
        exact_matches = np.flatnonzero(exact_boost > 0)
        if len(exact_matches):
            exact_order = exact_matches[
                np.argsort(-(scores[exact_matches] + exact_boost[exact_matches]))
            ]
            return [int(index) for index in exact_order[:count]]
        order = np.argsort(-(scores + exact_boost))
        return [int(index) for index in order[:count]]

    def _query_features(
        self, seed_indices: list[int], question_embedding: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        question_similarity = torch.tensor(
            self.graph.x.numpy() @ question_embedding, dtype=torch.float32
        )
        is_seed = torch.zeros(self.graph.num_nodes, dtype=torch.float32)
        is_seed[seed_indices] = 1.0
        node_features = torch.cat(
            [self.graph.x, question_similarity.unsqueeze(1), is_seed.unsqueeze(1)],
            dim=1,
        )
        relation_similarity = torch.tensor(
            self.relation_embeddings.numpy() @ question_embedding,
            dtype=torch.float32,
        ).unsqueeze(1)
        edge_similarity = relation_similarity.repeat(2, 1)
        edge_features = torch.cat([self.graph.edge_attr, edge_similarity], dim=1)
        return node_features, edge_features

    @staticmethod
    def _unique(values: list[int]) -> list[int]:
        result: list[int] = []
        seen: set[int] = set()
        for value in values:
            if value not in seen:
                result.append(value)
                seen.add(value)
        return result

    @torch.inference_mode()
    def retrieve(self, question: str, *, top_k: int = 12) -> dict[str, Any]:
        question_embedding = self._embed_question(question)
        seeds = self._find_seeds(question, question_embedding)
        node_features, edge_features = self._query_features(
            seeds, question_embedding
        )
        raw_scores = self.ranker(
            node_features.to(self.device),
            self.graph.edge_index.to(self.device),
            edge_features.to(self.device),
        ).cpu().numpy()

        neighbours: set[int] = set()
        for seed in seeds:
            neighbours.update(self.hop_one.get(seed, set()))
        ranked_neighbours = sorted(neighbours, key=lambda index: -raw_scores[index])
        global_order = [int(index) for index in np.argsort(-raw_scores)]
        order = self._unique(seeds + ranked_neighbours + global_order)[:top_k]

        candidates = []
        for rank, node_index in enumerate(order, start=1):
            row = self.entities.iloc[node_index]
            candidates.append(
                {
                    "rank": rank,
                    "source_id": f"E{node_index:04d}",
                    "title": str(row[self.title_col]),
                    "score": float(raw_scores[node_index]),
                    "seed": node_index in seeds,
                }
            )

        context, sources = self._build_context(order, seeds)
        return {
            "question": question,
            "seed_indices": seeds,
            "node_indices": order,
            "candidates": candidates,
            "context": context,
            "sources": sources,
        }

    def _build_context(
        self,
        node_indices: list[int],
        seed_indices: list[int],
        max_chars: int = 18_000,
    ) -> tuple[str, list[dict[str, str]]]:
        selected = set(node_indices)
        seeds = set(seed_indices)
        lines: list[str] = []
        sources: list[dict[str, str]] = []
        added_source_ids: set[str] = set()

        def add_source(source: dict[str, str]) -> None:
            if source["id"] not in added_source_ids:
                sources.append(source)
                added_source_ids.add(source["id"])

        direct_relationships: set[int] = set()
        for relation_index, relation in self.relationships.iterrows():
            source_idx = int(relation["source_idx"])
            target_idx = int(relation["target_idx"])
            if source_idx not in seeds and target_idx not in seeds:
                continue
            direct_relationships.add(int(relation_index))
            source_id = f"R{relation_index:04d}"
            source_title = str(self.entities.iloc[source_idx][self.title_col])
            target_title = str(self.entities.iloc[target_idx][self.title_col])
            description = str(relation.get("description", ""))[:1_000]
            lines.append(
                f"[{source_id}] {source_title} -> {target_title}: {description}"
            )
            add_source(
                {
                    "id": source_id,
                    "kind": "relationship",
                    "title": f"{source_title} → {target_title}",
                }
            )

        for node_index in node_indices:
            row = self.entities.iloc[node_index]
            source_id = f"E{node_index:04d}"
            title = str(row[self.title_col])
            description = str(row[self.text_col])[:1_200]
            lines.append(f"[{source_id}] {title}: {description}")
            add_source(
                {"id": source_id, "kind": "entity", "title": title}
            )

        for relation_index, relation in self.relationships.iterrows():
            source_idx = int(relation["source_idx"])
            target_idx = int(relation["target_idx"])
            if int(relation_index) in direct_relationships:
                continue
            if source_idx not in selected or target_idx not in selected:
                continue
            source_id = f"R{relation_index:04d}"
            source_title = str(self.entities.iloc[source_idx][self.title_col])
            target_title = str(self.entities.iloc[target_idx][self.title_col])
            description = str(relation.get("description", ""))[:1_000]
            lines.append(
                f"[{source_id}] {source_title} -> {target_title}: {description}"
            )
            add_source(
                {
                    "id": source_id,
                    "kind": "relationship",
                    "title": f"{source_title} → {target_title}",
                }
            )

        context = "\n".join(lines)
        return context[:max_chars], sources

    def _chat_completion(
        self, messages: list[dict[str, str]], *, max_tokens: int = 600
    ) -> str:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"

        last_error = ""
        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(5, 180),
                )
                if response.status_code in {429, 502, 503, 504} and attempt < 2:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError("LLM вернула пустой ответ")
                return content.strip()
            except (requests.RequestException, KeyError, ValueError) as error:
                response = getattr(error, "response", None)
                details = response.text[:500] if response is not None else str(error)
                last_error = details
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
        raise RuntimeError(
            f"Не удалось получить ответ от {self.llm_model} по "
            f"{self.llm_base_url}: {last_error}"
        )

    @staticmethod
    def _clean_answer(answer: str) -> str:
        answer = re.sub(r"[ \t]*\[(?:E|R)\d+\]", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"[ \t]+([,.;:!?])", r"\1", answer)
        answer = re.sub(r"([.!?])\1+", r"\1", answer)
        answer = re.sub(r"(?m)^\s*[.,;:]+\s*$", "", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer)
        return answer.strip()

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        top_k: int = 12,
    ) -> dict[str, Any]:
        retrieval = self.retrieve(question, top_k=top_k)
        recent_history = (history or [])[-6:]
        dialog = "\n".join(
            f"{item['role']}: {item['content']}" for item in recent_history
        )
        user_content = (
            f"КОНТЕКСТ ИЗ ГРАФА:\n{retrieval['context']}\n\n"
            f"ПРЕДЫДУЩИЙ ДИАЛОГ:\n{dialog or '(пусто)'}\n\n"
            f"ВОПРОС: {question}"
        )
        answer = self._chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Ты русскоязычный ассистент по базе знаний GraphRAG. "
                        "Отвечай только по переданному контексту. Дай ясный и "
                        "краткий ответ. Не показывай технические идентификаторы "
                        "источников вида [E0001] или [R0001]. Если данных недостаточно, "
                        "прямо скажи: «В графе недостаточно данных». Не выдумывай "
                        "источники и факты."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
        )
        retrieval["answer"] = self._clean_answer(answer)
        return retrieval

    def llm_health(self) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{self.llm_base_url}/models",
                headers={"Authorization": f"Bearer {self.llm_api_key}"},
                timeout=5,
            )
            response.raise_for_status()
            model_ids = [item.get("id") for item in response.json().get("data", [])]
            return {
                "ok": self.llm_model in model_ids,
                "models": model_ids,
                "message": (
                    "LLM подключена"
                    if self.llm_model in model_ids
                    else f"Модель {self.llm_model} не загружена"
                ),
            }
        except (requests.RequestException, ValueError) as error:
            return {"ok": False, "models": [], "message": str(error)}
