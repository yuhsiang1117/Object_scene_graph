"""Publish the background prompts' CLIP embeddings for Khronos's open-set object detector.

Khronos's InstanceForwarding detector drops a segment whose CLIP feature is closer than
`max_background_score` to a background prompt ("structure", "wall", "floor"). Its
RosEmbeddingGroup gets those embeddings either from the segmenter's `embed` service or, when the
config lists no prompts, as one FeatureVectors message on `<ns>/features`. In this environment the
service never becomes available to a client, though a minimal server of the same type does
(scratch probes, 2026-10-29), so this node takes the topic route with the same numbers:
semantic_inference's ClipWrapper (ViT-L/14, as the segmenter config names it) and its
`embed_text`, which is what the service answers with. Khronos subscribes with a volatile QoS,
so the message is re-published until the node is stopped.

    python background_embeddings.py [--topic /dualmap/background/features] [--model ViT-L/14]
"""
from __future__ import annotations

import argparse

PROMPTS = ("structure", "wall", "floor")


def embeddings(model_name: str):
    import torch
    from semantic_inference.models.wrappers import ClipConfig, ClipWrapper

    config = ClipConfig()
    config.model_name = model_name
    encoder = ClipWrapper(config).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return [encoder.embed_text(p).float().cpu().numpy().squeeze() for p in PROMPTS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/dualmap/background/features")
    ap.add_argument("--model", default="ViT-L/14")
    ap.add_argument("--period", type=float, default=0.5)
    args, _ = ap.parse_known_args()

    vectors = embeddings(args.model)
    import rclpy
    from semantic_inference_msgs.msg import FeatureVector, FeatureVectors

    rclpy.init()
    node = rclpy.create_node("dualmap_background_embeddings")
    pub = node.create_publisher(FeatureVectors, args.topic, 1)
    msg = FeatureVectors(names=list(PROMPTS),
                         features=[FeatureVector(data=v.astype(float).tolist()) for v in vectors])
    node.create_timer(args.period, lambda: pub.publish(msg))
    node.get_logger().info(f"publishing {len(vectors)} background embeddings "
                           f"({vectors[0].shape[0]}-d, {args.model}) on {args.topic}")
    rclpy.spin(node)


if __name__ == "__main__":
    main()
