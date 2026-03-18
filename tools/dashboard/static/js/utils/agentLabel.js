/**
 * agentLabel.js
 *
 * Pure utility for resolving agent role keys to display names
 * using the manifest's agent_profiles map.
 */

/**
 * Build an agent label resolver from a manifest's profile map.
 * Returns a function that maps role keys (e.g. "value") to
 * profile names (e.g. "value_enriched").
 */
export function makeAgentLabel(manifest) {
  let m = manifest !== undefined && manifest !== null ? manifest : {};
  let profileMap = (m.agent_profiles && typeof m.agent_profiles === 'object') ? m.agent_profiles : null;
  /** Resolve an agent role key to its display name from the profile map. */
  return function agentLabel(role) {
    if (profileMap !== null && profileMap[role] !== undefined) {
      let val = profileMap[role];
      // agent_profiles may be {role: "profile_name"} (simple) or
      // {role: {system_prompts: ..., user_prompts: ...}} (full config).
      // Only use string values as display names.
      if (typeof val === 'string') return val;
    }
    return role;
  };
}
