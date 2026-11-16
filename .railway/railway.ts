// Railway configuration of the development/preview service (Infrastructure as Code). It replaces `railway.toml`,
// which Railway stops reading on 2026-12-01. Railway is not a bank deployment path (none is maintained; see
// docs/DEPLOYMENT.md#bank-deployment). The corpus file is built into the image, so the service needs no database.
//
// Variables are not managed here (`variables: { managed: false }`): they stay in Railway and never enter source.
// The container binds 0.0.0.0, so the fail-closed HTTP policy refuses startup until the service supplies the
// complete non-loopback policy as variables:
//   BDDK_HTTP_ALLOWED_HOSTS        exact public hostname of the Railway domain
//   BDDK_HTTP_ALLOWED_ORIGINS      exact HTTPS origins allowed to call /mcp
//   and either the full BDDK_JWT_* set or BDDK_HTTP_ALLOW_UNAUTHENTICATED=true
//     (unauthenticated exposure on a public Railway URL serves the corpus to the whole Internet; prefer JWT or a
//     private network).
//
// Preview a change with `railway config plan`; nothing reaches the service before `railway config apply`.
import { defineRailway, github, project, service } from "railway/iac";

// This repository manages only its own resources in the environment. Other
// repositories export their own partial name.
// See https://docs.railway.com/infrastructure-as-code#multi-repo-projects
export const partial = "bddk-mcp";

export default defineRailway(() => {
  const bddkMcp = service("bddk-mcp", {
    // checkSuites: a deployment waits for the commit's check suites (CI) and is skipped when one fails.
    source: github("omercagatay/bddk-mcp", { branch: "main", checkSuites: true }),
    build: { buildEnvironment: "V3", builder: "dockerfile", dockerfilePath: "Dockerfile" },
    healthcheck: "/health/ready",
    healthcheckTimeout: 300,
    deploy: { restartPolicyType: "ON_FAILURE", restartPolicyMaxRetries: 3 },
    replicas: { "europe-west4-drams3a": 1 },
    domains: ["mcp.bankreg.app"],
  });

  return project("perceptive-freedom", {
    variables: { managed: false },
    resources: [bddkMcp],
  });
});
