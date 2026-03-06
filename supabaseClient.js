import { createClient } from '@supabase/supabase-js'

// Replace with your actual values:
const supabaseUrl = 'https://supabase.com/dashboard/project/aocwwdntjckckcmlolet/settings/api-keys'  // Get this from Project Settings
const supabaseAnonKey = 'sb_publishable_93NHLdsZNdl9S6VzeelA8w_Hf2ES3aS'  // Use the published key!

export const supabase = createClient(supabaseUrl, supabaseAnonKey)